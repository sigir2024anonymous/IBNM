# --------------------------------------------------------
# Reference from https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------
# from .swin_transformer import SwinTransformer
# from .swin_transformer_BCAT import SwinTransformerBCAT
# from .vit import VisionTransformer
# from .vit_2 import VisionTransformer as vit2
# from .vit_BCAT import VisionTransformer as VisionTransformerBCAT
from .vit import PMTrans


def build_model(config, logger=None):
    model_type = config.MODEL.TYPE
    num_class = config.MODEL.NUM_CLASSES
    if model_type == 'swin':
        model = PMTrans(num_classes=num_class)
    elif model_type == 'deit':
        model = PMTrans(num_classes=num_class, model_name='deit_base')
    elif model_type == 'vit':
        model = PMTrans(num_classes=num_class, model_name='vit_base')    
        
    if config.MODEL.RESUME:
        model.load_pretrained(checkpoint_path=config.MODEL.RESUME, logger=logger)  
          
    return model
