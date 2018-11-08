using Neuro.Tensors;
using System;

namespace DeepQL
{
    public static partial class TensorExtensions
    {
        public static void FillWithPixelData(this Tensor t, byte[] pixelData)
        {
            bool grayscale = t.Shape.Depth == 1;
            
            if (!grayscale && t.Shape.Depth != 3)
                throw new Exception("Invalid tensor depth. Expected depth 3 for RGB.");

            if (pixelData.Length != t.Shape.Length)
                throw new Exception($"Invalid tensor length. Expected room for {pixelData.Length} values.");

            int COLORS_PER_PIXEL = t.Shape.Depth;
            int PIXELS_PER_COLOR = pixelData.Length / COLORS_PER_PIXEL;

            for (int offset = 0; offset < COLORS_PER_PIXEL; ++offset)
            for (int i = 0; i < PIXELS_PER_COLOR; ++i)
                t.SetFlat(pixelData[i * COLORS_PER_PIXEL + offset], offset * PIXELS_PER_COLOR + i);
        }
    }
}
