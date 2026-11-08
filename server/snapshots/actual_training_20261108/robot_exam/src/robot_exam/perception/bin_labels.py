"""Versioned upright printed category labels; no change to physical geometry."""
from pathlib import Path

import numpy as np
from PIL import Image


def upright_label_textures(root):
    changes=[]
    for category in ('apple','banana','orange'):
        texture=root.find(f"./asset/texture[@name='label_{category}']")
        original=Path(texture.get('file'))
        destination=original.with_name(category+'_upright_v2.png')
        with Image.open(original) as image:
            fixed=image.transpose(Image.Transpose.ROTATE_180)
            if destination.exists():
                with Image.open(destination) as existing:
                    if not np.array_equal(np.array(existing),np.array(fixed)):
                        raise ValueError('Preserve conflicting label asset: '+str(destination))
            else:fixed.save(destination)
        texture.set('file',str(destination))
        changes.append(dict(category=category,source=str(original),file=str(destination),pixel_transform='rotate180'))
    return changes
