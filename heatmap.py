import torch
from PIL import Image

from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,
)
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.models import ColQwen2, ColQwen2Processor

from colpali_engine.utils.torch_utils import get_torch_device

model_name = "./models/colqwen2-v1.0-merged"
device = get_torch_device("auto")

# Load the model
model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()

# Load the processor
processor = ColQwen2Processor.from_pretrained(model_name)

# Load the image and query
image = Image.open("test_data/cpu.jpg")
query = "Is AMD's CPU performance better than Intel's?"

# Preprocess inputs
batch_images = processor.process_images([image]).to(device)
batch_queries = processor.process_queries([query]).to(device)

# Forward passes
with torch.no_grad():
    image_embeddings = model.forward(**batch_images)
    query_embeddings = model.forward(**batch_queries)

# Get the number of image patches
n_patches = processor.get_n_patches(image_size=image.size, spatial_merge_size=model.spatial_merge_size)

# Get the tensor mask to filter out the embeddings that are not related to the image
image_mask = processor.get_image_mask(batch_images)

# Generate the similarity maps
batched_similarity_maps = get_similarity_maps_from_embeddings(
    image_embeddings=image_embeddings,
    query_embeddings=query_embeddings,
    n_patches=n_patches,
    image_mask=image_mask,
)

# Get the similarity map for our (only) input image
similarity_maps = batched_similarity_maps[0]  # (query_length, n_patches_x, n_patches_y)

# Tokenize the query
query_tokens = processor.tokenizer.tokenize(query)
print(query_tokens)
print(similarity_maps.shape)

# Plot and save the similarity maps for each query token
plots = plot_all_similarity_maps(
    image=image,
    query_tokens=query_tokens,
    similarity_maps=similarity_maps,
    add_title=True,
    show_colorbar=True,
)


for idx, (fig, ax) in enumerate(plots):
    fig.savefig(f"test_output/similarity_map_{idx}.png")