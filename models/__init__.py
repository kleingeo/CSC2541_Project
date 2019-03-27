def create_model(opt):
    model = None
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned' or
               opt.dataset_mode == 'unaligned3dtoimage' or
               opt.dataset_mode == 'unaligned2dtoimage')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned' or
               opt.dataset_mode == 'aligned3d' or
               opt.dataset_mode == 'aligned3dto2d' or
               opt.dataset_mode == 'aligned3dtoimage' or
               opt.dataset_mode == 'aligned2dtoimage' or
               opt.dataset_mode == 'aligned2dnpy')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()

    elif opt.model == 'pix2pix_uncertainty':
        assert(opt.dataset_mode == 'aligned' or
               opt.dataset_mode == 'aligned3d' or
               opt.dataset_mode == 'aligned3dto2d' or
               opt.dataset_mode == 'aligned3dtoimage' or
               opt.dataset_mode == 'aligned2dtoimage' or
               opt.dataset_mode == 'aligned2dnpy')
        from .pix2pix_uncertainty_model import Pix2PixUncertaintyModel
        model = Pix2PixUncertaintyModel()

    elif opt.model == 'pix2pix_uncertainty_net':
        assert(opt.dataset_mode == 'aligned' or
               opt.dataset_mode == 'aligned3d' or
               opt.dataset_mode == 'aligned3dto2d' or
               opt.dataset_mode == 'aligned3dtoimage' or
               opt.dataset_mode == 'aligned2dtoimage' or
               opt.dataset_mode == 'aligned2dnpy')
        from .pix2pix_uncertainty_net_model import Pix2PixUncertaintyNetModel
        model = Pix2PixUncertaintyNetModel()

    elif opt.model == 'reg_unc_net':
        assert(opt.dataset_mode == 'aligned' or
               opt.dataset_mode == 'aligned3d' or
               opt.dataset_mode == 'aligned3dto2d' or
               opt.dataset_mode == 'aligned3dtoimage' or
               opt.dataset_mode == 'aligned2dtoimage' or
               opt.dataset_mode == 'aligned2dnpy')
        from .reg_unc_model import RegressionUncertaintyNetModel
        model = RegressionUncertaintyNetModel()



    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
