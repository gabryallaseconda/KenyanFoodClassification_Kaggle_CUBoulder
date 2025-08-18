import albumentations

from neural_network.configuration import dataConfig

resize_pipeline = albumentations.Compose([
        albumentations.LongestMaxSize(max_size=320),
        albumentations.PadIfNeeded(min_height=320, 
                                   min_width=320, 
                                   border_mode=0, 
                                   fill=255),
        albumentations.Resize(height=dataConfig.resized_image_height, 
                              width=dataConfig.resized_image_width)  
        ])
    

def get_resize_pipeline():
    return resize_pipeline

def get_training_augmentation_pipeline():
    return  albumentations.Compose([
    
        # Resize
        resize_pipeline,
            
        # Noise or blur
        albumentations.OneOf([
            albumentations.AdditiveNoise(
                noise_type="gaussian",
                spatial_mode="shared",
                noise_params={"mean_range":[0,0],"std_range":[0.05,0.15]},
                approximation=1
            ), 
            albumentations.Blur(
                blur_limit=[3, 7]
            ),
            albumentations.Defocus(
                radius=[3, 6],
                alias_blur=[0.1, 0.5]
            ),
            albumentations.GaussNoise(
                std_range=[0.1, 0.2],
                mean_range=[0, 0],
                per_channel=True,
                noise_scale_factor=0.2
            ),
            albumentations.GlassBlur(
                sigma=0.3,
                max_delta=3,
                iterations=2,
                mode="fast"
            ),

            albumentations.MotionBlur(
                blur_limit=[5, 11],
                allow_shifted=False,
                angle_range=[0, 90],
                direction_range=[-1, 1]
            )
        ], p = 0.75),
        
        # Flips
        albumentations.HorizontalFlip(p=0.5),
        
        # HSL adjustments
        albumentations.OneOf([
            albumentations.CLAHE(
                clip_limit=4,
                tile_grid_size=[8, 8]
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=[-0.3, 0.3],
                contrast_limit=[-0.3, 0.3],
                brightness_by_max=True,
                ensure_safe_range=False
            ),
            albumentations.ColorJitter(
                brightness=[0.8, 1.2],
                contrast=[0.8, 1.2],
                saturation=[0.4, 0.6],
                hue=[-0.05, 0.05]
            )
        ], p = .75)
    ])    
    


