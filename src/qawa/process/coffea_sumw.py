from coffea import processor
import awkward as ak


class coffea_sumw(processor.ProcessorABC):
    def __init__(self):
        super().__init__()

    def process(self, event: processor.LazyDataFrame):
        dataset_name = event.metadata['dataset']
        is_data = event.metadata.get("is_data")

        sumw = 1.0
        if is_data:
            sumw = -1.0
        else:
            sumw = ak.sum(event.genEventSumw)

        return {dataset_name: sumw}

    def postprocess(self, accumulator):
        return accumulator
