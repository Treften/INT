# common
import logging

from multiprocessing.queues import Queue

import torch

from pytorch.common.abstract.abstract_process_worker import AbstractProcessWorker
from pytorch.common.batcher.batch_primitives import Batch

logger = logging.getLogger("root")


class BatchDiskReader(AbstractProcessWorker):
    def __init__(self, in_queue, out_queue, image_processor, audio_processor):
        AbstractProcessWorker.__init__(self)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.image_processor = image_processor
        self.audio_processor = audio_processor

    def do_work_once(self):
        #print("READER WAIT", self.pid, flush=True)
        try:
            input_batch = self.in_queue.get(timeout=1)
        except Exception:
            return
    #     print(
    #     "READER GOT TASK",
    #     self.pid,
    #     len(input_batch.data_samples),
    #     flush=True
    # )
        try:
            out_samples = []

            for data_sample in input_batch.data_samples:
                try:
                    if self.image_processor is not None:
                        image = self.image_processor.process(data_sample.img_rel_path)
                        if image is None:
                            logger.error("Cannot load image: %s", data_sample.img_rel_path)
                            continue
                        data_sample.image = image

                    if self.audio_processor is not None:
                        spec = self.audio_processor.process(data_sample.wav_rel_path)
                        if spec is None:
                            logger.error("Cannot load wav: %s", data_sample.wav_rel_path)
                            continue
                        data_sample.image = spec

                    out_samples.append(data_sample)

                except Exception:
                    logger.exception("BatchDiskReader: sample processing error")

            if self.audio_processor is not None and out_samples:
                max_len = max(x.image.size(1) for x in out_samples) + 1
                hspec = out_samples[0].image.size(0)

                for x in out_samples:
                    spec_padded = torch.zeros(hspec, max_len)
                    spec_padded[:, :x.image.size(1)] = x.image
                    x.image = spec_padded
            #print("READER PUT", self.pid, flush=True)
            self.out_queue.put(Batch(out_samples))
            #print("READER PUT DONE", self.pid, flush=True)

        except Exception:
            logger.exception("BatchDiskReader: batch processing error")
            self.out_queue.put(Batch([]))

        finally:
            self.in_queue.task_done()