class InputPar:
    def __init__(self, property_str, img_size, img_chan, n_class, n_per_label,
                 percent=None, alabel=None, mr_name=None, mr_par=None):
        self.type = property_str
        self.img_size = img_size
        self.img_chan = img_chan
        self.class_num = n_class

        self.n_per_label = n_per_label
        self.percent = percent
        self.alabel = alabel
        self.mr_name = mr_name
        self.mr_par = mr_par
        self.n_follow_up_samples = 0

