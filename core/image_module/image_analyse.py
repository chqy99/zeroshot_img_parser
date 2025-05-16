import abc

class ImageAnalyse(abc.ABC):
    @abc.abstractmethod
    def analyse_ocr(self, image, **kwargs):
        pass

    @abc.abstractmethod
    def analyse_edge(self, image, **kwargs):
        pass

    @abc.abstractmethod
    def analyse_one_instance(self, image, **kwargs):
        pass

    @abc.abstractmethod
    def analyse_instance_segm(self, image, **kwargs):
        pass

    @abc.abstractmethod
    def analyse_multi_method(self, image, **kwargs):
        pass
