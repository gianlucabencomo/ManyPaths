import numpy as np
import matplotlib.pyplot as plt


def generate_concept(bits, scale: float = 1.0):
    if not (len(bits) == 4):
        raise ValueError("Bits must be length 4.")

    # Initialize a blank grid
    grid_image = np.ones((32, 32, 3), dtype=np.uint8) * 255

    # Extract tens and units digits
    color = (1, 2) if bits[0] == 1 else (0, 1)
    shape = bits[1] == 1
    size = 4 if bits[2] == 1 else 10
    style = bits[3] == 1

    if shape:
        grid_image[size:32-size,size:32-size,color] = 0
        if style == 1:
            grid_image[size:32-size,size:32-size:2,color] = 200
    else:
        for i in range(32 - 2 * size):
            grid_image[32 - (size + i + 1),i // 2 + size:32 - i // 2 - size,color] = 0
        if style == 1:
            for i in range(0, 32, 1):
                for j in range(0, 32, 2):
                    if grid_image[i,j,color].any() == 0:
                        grid_image[i,j,color] = 200
    grid_image = grid_image / scale
    return grid_image

def plot():
    fig, axes = plt.subplots(2, 8, figsize=(14, 5))
    for i in range(16):
        bits = [int(x) for x in f"{i:04b}"]
        grid_image = generate_concept(bits)
        ax = axes[i // 8, i % 8]
        ax.imshow(grid_image, interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"{bits[0]}{bits[1]}{bits[2]}{bits[3]}", fontsize=20)

    plt.tight_layout(pad=0)
    output_path = "concept_grid.pdf"
    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    plot()
