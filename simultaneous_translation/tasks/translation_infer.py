# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .waitk_sequence_generator import WaitkSequenceGenerator
from .inference_config import InferenceConfig
import logging
import torch
import numpy as np
from fairseq import metrics, utils
from fairseq.tasks import register_task, LegacyFairseqTask
from fairseq.tasks.translation import TranslationTask
from fairseq.logging.meters import safe_round

from fairseq.scoring.bleu import SacrebleuScorer
from fairseq.scoring.wer import WerScorer

logger = logging.getLogger(__name__)

EVAL_BLEU_ORDER = 4


@register_task("translation_infer")
class TranslationWInferenceTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)

        parser.add_argument(
            "--inference-config-yaml",
            type=str,
            default="inference.yaml",
            help="Configuration YAML filename for bleu or wer eval (under exp/)",
        )

        parser.add_argument(
            "--from-encoder", action="store_true",
            help=(
                'decode prediction from model.encoder.'
            ),
        )

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        # for waitk models, left padding is bad for inference.
        self.cfg.left_pad_source = False

        self.inference_cfg = InferenceConfig(args.inference_config_yaml)
        self.pre_tokenizer = self.build_tokenizer(args)

        self.from_encoder = getattr(args, "from_encoder", False)

    def build_model(self, args):
        model = super().build_model(args)
        if self.inference_cfg.eval_any:
            self.sequence_generator = self.build_generator(
                [model],
                self.inference_cfg.generation_args,
            )
        return model

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        """ speech_to_text ignores seq_gen_cls and overrides
        extra_gen_cls_kwargs. So we will call LegacyFairseqTask's
        method. """
        waitk = getattr(models[0], "waitk", None)
        test_waitk = getattr(self.inference_cfg.generation_args, "waitk", None)
        if test_waitk is not None and test_waitk != waitk:
            # test override.
            logger.warning(
                f"Train test mismatch: training wait-{waitk}, while testing wait-{test_waitk}.")
            waitk = test_waitk
        pre_ratio = 1
        if waitk is not None:
            pre_ratio = models[0].pre_decision_ratio
            seq_gen_cls = WaitkSequenceGenerator
            extra = {"waitk": waitk, "pre_decision_ratio": pre_ratio}
            if extra_gen_cls_kwargs:
                extra_gen_cls_kwargs.update(extra)
            else:
                extra_gen_cls_kwargs = extra
        return LegacyFairseqTask.build_generator(
            self, models, args, seq_gen_cls=seq_gen_cls, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.inference_cfg.eval_any:
            _metrics = self._inference_with_metrics(self.sequence_generator, sample, model)

        if self.inference_cfg.eval_bleu:
            bleu = _metrics["bleu"]
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        if self.inference_cfg.eval_wer:
            logging_output.update(_metrics["wer"])
        return loss, sample_size, logging_output

    @torch.no_grad()
    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        if getattr(models[0], "one_pass_decoding", False):
            # one-pass decoding
            if hasattr(self, 'blank_symbol'):
                sample["net_input"]["blank_idx"] = self.tgt_dict.index(self.blank_symbol)
            sample["net_input"]["from_encoder"] = self.from_encoder
            return models[0].generate(**sample["net_input"])
        else:
            # incremental decoding
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )

    def _inference_with_metrics(self, generator, sample, model):

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.inference_cfg.post_process,  # this will handle bpe for us.
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.pre_tokenizer is not None:
                s = self.pre_tokenizer.decode(s)
            return s if s else "UNKNOWNTOKENINHYP"

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(
                decode(gen_out[i][0]["tokens"])
            )
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.inference_cfg.print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])

        ret = {}
        if self.inference_cfg.eval_bleu:
            bleu_scorer = SacrebleuScorer(self.inference_cfg.eval_bleu_args)
            for h, r in zip(hyps, refs):
                bleu_scorer.add_string(ref=r, pred=h)

            ret["bleu"] = bleu_scorer.sacrebleu.corpus_bleu(
                bleu_scorer.pred, [bleu_scorer.ref],
                tokenize="none"  # use none because it's handled by SacrebleuScorer
            )

        if self.inference_cfg.eval_wer:
            wer_scorer = WerScorer(self.inference_cfg.eval_wer_args)
            for h, r in zip(hyps, refs):
                wer_scorer.add_string(ref=r, pred=h)

            ret["wer"] = {
                "wv_errors": wer_scorer.distance,
                "w_errors": wer_scorer.distance,
                "w_total": wer_scorer.ref_length
            }

        return ret

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        if self.inference_cfg.eval_bleu:

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

        if self.inference_cfg.eval_wer:

            w_errors = sum_logs("w_errors")
            wv_errors = sum_logs("wv_errors")
            w_total = sum_logs("w_total")

            metrics.log_scalar("_w_errors", w_errors)
            metrics.log_scalar("_wv_errors", wv_errors)
            metrics.log_scalar("_w_total", w_total)

            if w_total > 0:
                metrics.log_derived(
                    "wer",
                    lambda meters: safe_round(
                        meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                    )
                    if meters["_w_total"].sum > 0
                    else float("nan"),
                )
                metrics.log_derived(
                    "raw_wer",
                    lambda meters: safe_round(
                        meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                    )
                    if meters["_w_total"].sum > 0
                    else float("nan"),
                )
