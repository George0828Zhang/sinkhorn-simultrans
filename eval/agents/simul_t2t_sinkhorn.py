# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from torchinfo import summary
import os
import math
import logging
from fairseq import checkpoint_utils, tasks, utils
import sentencepiece as spm
import torch

try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import TextAgent
    from simuleval.states import ListEntry, TextStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
BOW_PREFIX = "\u2581"


class SimulTransTextAgentSinkhorn(TextAgent):
    """
    Simultaneous Translation
    Text agent for sinkhorn encoder models
    """
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        # parser.add_argument("--max-len", type=int, default=100,
        #                     help="Max length of translation")
        parser.add_argument("--tgt-splitter-type", type=str, default="SentencePiece",
                            help="Subword splitter type for target text.")
        parser.add_argument("--tgt-splitter-path", type=str, default=None,
                            help="Subword splitter model path for target text.")
        parser.add_argument("--src-splitter-type", type=str, default="SentencePiece",
                            help="Subword splitter type for source text.")
        parser.add_argument("--src-splitter-path", type=str, default=None,
                            help="Subword splitter model path for source text.")
        parser.add_argument("--user-dir", type=str, default="examples/simultaneous_translation",
                            help="User directory for simultaneous translation")
        parser.add_argument("--max-len-a", type=int, default=1.2,
                            help="Max length of translation ax+b")
        parser.add_argument("--max-len-b", type=int, default=10,
                            help="Max length of translation ax+b")
        parser.add_argument("--force-finish", default=False, action="store_true",
                            help="Force the model to finish the hypothsis if the source is not finished")
        parser.add_argument("--test-waitk", type=int, default=1)
        parser.add_argument("--incremental-encoder", default=False, action="store_true",
                            help="Update the model incrementally without recomputation of history.")
        parser.add_argument("--print-blank", default=False, action="store_true",
                            help="for debug purpose.")

        # fmt: on
        return parser

    def __init__(self, args):
        self.args = args

        self.force_finish = args.force_finish
        self.incremental_encoder = args.incremental_encoder
        self.print_blank = args.print_blank

        # Whether use gpu
        self.gpu = getattr(args, "gpu", False)

        # Load Model
        self.load_model_vocab(args)

        # build word splitter
        self.build_word_splitter(args)

        self.eos = DEFAULT_EOS

        # Max len
        self.max_len = lambda x: self.model.encoder.upsample_ratio * x

        torch.set_grad_enabled(False)

    def load_model_vocab(self, args):
        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(
            path=filename, arg_overrides=None, load_on_all_ranks=False)

        cfg = state["cfg"]

        # update overwrites:
        cfg.common.user_dir = args.user_dir
        cfg.task.data = args.data_bin
        cfg.model.load_pretrained_encoder_from = None
        cfg.model.load_pretrained_decoder_from = None

        utils.import_user_module(cfg.common)
        # Setup task, e.g., translation, language modeling, etc.
        task = tasks.setup_task(cfg.task)
        # Build model and criterion
        model = task.build_model(cfg.model)

        model.load_state_dict(
            state["model"], strict=True, model_cfg=cfg.model
        )

        # Optimize ensemble for generation
        if self.gpu:
            model.cuda()
        model.prepare_for_inference_(cfg)

        self.model = model

        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary
        self.dict["src"] = task.source_dictionary

        self.blank_idx = self.dict["tgt"].bos()

        # remove bos index so that it will be printed by dict.string function
        if self.print_blank:
            delattr(self.dict["tgt"], "bos_index")

        self.pre_tokenizer = task.pre_tokenizer

        # logger.info(summary(self.model))
        logger.info("task: {}".format(task.__class__.__name__))
        logger.info("model: {}".format(self.model.__class__.__name__))
        logger.info("pre_tokenizer: {}".format(self.pre_tokenizer))

    def build_word_splitter(self, args):
        self.spm = {}
        for lang in ['src', 'tgt']:
            if getattr(args, f'{lang}_splitter_type', None):
                path = getattr(args, f'{lang}_splitter_path', None)
                if path:
                    self.spm[lang] = spm.SentencePieceProcessor()
                    self.spm[lang].Load(path)

    def initialize_states(self, states):
        states.units.source = ListEntry()
        states.units.target = ListEntry()
        states.enc_incremental_states = dict()
        states.last_token_index = self.blank_idx

    def build_states(self, args, client, sentence_id):
        # Initialize states here, for example add customized entry to states
        # This function will be called at beginning of every new sentence
        states = TextStates(args, client, sentence_id, self)
        self.initialize_states(states)
        return states

    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

    def segment_to_units(self, segment, states):
        # Split a full word (segment) into subwords (units)
        return self.spm['src'].EncodeAsPieces(segment)

    def forward_encoder(self, *args, **kwargs):
        return self.model.encoder.causal_encoder(*args, **kwargs)

    def forward_decoder(self, encoder_out, dec_len=1):
        x = encoder_out["encoder_out"][0][-dec_len:]
        x, _ = self.model.encoder.upsample(x, None)
        x = self.model.output_projection(x)
        x = x.transpose(1, 0)  # force batch first
        return x, None

    def update_model_encoder(self, states, finish=False):
        delay = self.args.test_waitk
        src_len = len(states.units.source)
        enc_len = 0

        if getattr(states, "encoder_states", None) is not None:
            enc_len = states.encoder_states["encoder_out"][0].size(0)

        if not finish and src_len < enc_len + delay:
            return

        logger.info("ENC")
        src_indices = [
            self.dict['src'].index(x)
            for x in states.units.source.value
        ]

        if states.finish_read() and src_indices[-1] != self.dict["tgt"].eos_index:
            # Append the eos index when the prediction is over
            src_indices += [self.dict["tgt"].eos_index]

        src_indices = self.to_device(
            torch.LongTensor(src_indices).unsqueeze(0)
        )
        src_lengths = self.to_device(
            torch.LongTensor([src_indices.size(1)])
        )

        encoder_out = self.forward_encoder(
            src_indices,
            src_lengths,
            incremental_state=states.enc_incremental_states,
            incremental_step=src_len - enc_len,
        )

        # pruning incomplete encoding output and states
        encoder_out["encoder_out"][0] = encoder_out["encoder_out"][0][:1]
        self.model.encoder.causal_encoder.clear_cache(
            states.enc_incremental_states,
            keep=enc_len + 1
        )

        if getattr(states, "encoder_states", None) is None:
            states.encoder_states = {
                "encoder_out": encoder_out["encoder_out"],  # List[T x B x C]
                "encoder_padding_mask": [],  # B x T
                "encoder_embedding": [],  # B x T x C
                "encoder_states": [],  # List[T x B x C]
                "src_tokens": [],
                "src_lengths": [],
            }
        else:
            states.encoder_states["encoder_out"][0] = torch.cat(
                (
                    states.encoder_states["encoder_out"][0],
                    encoder_out["encoder_out"][0]
                ), dim=0
            )

        # logger.info(states.encoder_states["encoder_out"][0].shape)

        torch.cuda.empty_cache()

    def update_states_read(self, states):
        # Happens after a read action.
        self.update_model_encoder(states)

    def units_to_segment(self, unit_queue, states):
        """
        queue: stores bpe tokens.
        server: accept words.

        Therefore, we need merge subwords into word. we find the first
        subword that starts with BOW_PREFIX, then merge with subwords
        prior to this subword, remove them from queue, send to server.
        """
        # Merge sub word to full word.
        tgt_dict = self.dict["tgt"]

        debug_q = [tgt_dict.string([u]) for u in unit_queue]

        # if segment starts with eos, send EOS
        if tgt_dict.eos() == unit_queue[0]:
            return DEFAULT_EOS

        string_to_return = None
        last_token_index = states.last_token_index

        def decode(tok_idx):
            toks = torch.LongTensor([last_token_index] + list(tok_idx))
            toks = toks.unique_consecutive()
            if toks.size(-1) == 1:
                # all tokens equals last token
                return None
            toks = toks[1:]
            if toks.eq(self.blank_idx).all():
                toks = toks[-1:]
            else:
                toks = toks[toks != self.blank_idx]

            states.last_token_index = toks[-1]

            hyp = tgt_dict.string(
                toks,
                "sentencepiece",
            )
            if self.pre_tokenizer is not None:
                hyp = self.pre_tokenizer.decode(hyp)
            return hyp

        # if force finish, there will be None's
        segment = []
        if None in unit_queue.value:
            unit_queue.value.remove(None)

        src_len = states.encoder_states["encoder_out"][0].size(0)
        if (
            (len(unit_queue) > 0 and tgt_dict.eos() == unit_queue[-1])
            or len(states.units.target) > self.max_len(src_len)
        ):
            hyp = decode(unit_queue)
            string_to_return = ([hyp] if hyp else []) + [DEFAULT_EOS]
        else:
            space_p = None
            ignore = [last_token_index]
            for p, unit_id in enumerate(unit_queue):
                if p == 0:
                    ignore += [unit_id]
                token = tgt_dict.string([unit_id])
                if (
                    token.startswith(BOW_PREFIX)
                    and unit_id not in ignore
                ):
                    """
                    find the first tokens with escape symbol 
                    that is not a continuation of previous tokens
                    """
                    space_p = p
                    break
            if space_p is not None:
                for j in range(space_p):
                    segment += [unit_queue.pop()]

                hyp = decode(segment)
                string_to_return = [hyp] if hyp else []

                if tgt_dict.eos() == unit_queue[0]:
                    string_to_return += [DEFAULT_EOS]

        logger.debug(f"QUEUE: {debug_q},\tSTRING: {string_to_return}")

        return string_to_return

    def policy(self, states):
        if getattr(states, "decoder_out", None) is not None:
            return WRITE_ACTION

        if not states.finish_read():
            return READ_ACTION
        else:
            src_indices = [
                self.dict['src'].index(x)
                for x in states.units.source.value
            ]
            src_indices += [self.dict["tgt"].eos_index]
            src_indices = self.to_device(
                torch.LongTensor(src_indices).unsqueeze(0)
            )
            src_lengths = self.to_device(
                torch.LongTensor([src_indices.size(1)])
            )

            encoder_out = self.forward_encoder(
                src_indices,
                src_lengths,
            )

            states.encoder_states = encoder_out

            logits, _ = self.forward_decoder(
                encoder_out=encoder_out,
                dec_len=0
            )
            _eos = logits.new_zeros((1, 1, logits.size(-1)))
            _eos[...,self.dict["tgt"].eos_index:self.dict["tgt"].eos_index+1] = 1
            logits = torch.cat(
                (
                    logits,
                    _eos
                ),
                dim=1
            )
            states.decoder_out = logits

            # lprobs = self.model.get_normalized_probs(
            #     [logits], log_probs=True
            # )
            # debug_1 = lprobs.argmax(dim=-1).squeeze(0)
            # debug_2 = debug_1.unique_consecutive()
            # debug_3 = debug_2[debug_2 != 0]
            # debug_4 = self.dict["tgt"].string(debug_3)

            # import pdb; pdb.set_trace()

            torch.cuda.empty_cache()

            return WRITE_ACTION

    def predict(self, states):

        # logger.debug("WRITE")
        lprobs = self.model.get_normalized_probs(
            [states.decoder_out[:, :1]], log_probs=True
        )

        index = lprobs.argmax(dim=-1)

        index = index[0, 0].item()

        if states.decoder_out.size(1) < 2:
            states.decoder_out = None
        else:
            states.decoder_out = states.decoder_out[:, 1:]

        if (
            self.force_finish
            and index == self.dict["tgt"].eos()
            and not states.finish_read()
        ):
            # If we want to force finish the translation
            # (don't stop before finish reading), return a None
            self.model.decoder.clear_cache(states.dec_incremental_states)
            index = None

        logger.info(f"({index})")
        return index
