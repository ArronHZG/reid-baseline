class Data:
    def __init__(self):
        self.feat_x = None
        self.feat_t = None
        self.feat_c = None
        self.cls_score = None
        self.cls_label = None

        self.recon_ae = None  # feature
        self.ae = None  # model

        self.source = None

    def __str__(self):
        print(self.feat_t)
        print(self.feat_c)
        print(self.cls_score)
        print(self.cls_label)
        return ""
