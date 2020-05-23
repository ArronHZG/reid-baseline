class Data:
    def __init__(self):
        self.feat_t = None
        self.feat_c = None
        self.cls_score = None
        self.cls_label = None
        self.source_feat_c = None
        self.source_feat_t = None
        self.feat_t_decode = None
        self.recon_ae = None
        self.recon_ael1 = None
        self.recon_ael2 = None
        self.recon_cae = None
        self.ael1 = None
        self.ael2 = None
        self.cae = None

    def __str__(self):
        print(self.feat_t)
        print(self.feat_c)
        print(self.cls_score)
        print(self.cls_label)
        return ""
