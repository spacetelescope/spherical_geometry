import os

def resolve_imagename(ROOT_DIR, base_name):
    """Resolve image name for tests."""

    image_name = os.path.join(ROOT_DIR, base_name)

    # Is it zipped?
    if not os.path.exists(image_name):
        image_name = image_name.replace('.fits', '.fits.gz')

    return image_name
