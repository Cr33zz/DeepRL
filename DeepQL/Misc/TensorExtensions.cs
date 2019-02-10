using Neuro.Tensors;
using System;

namespace DeepQL
{
    public static partial class TensorExtensions
    {
        public static void FillWithPixelData(this Tensor t, byte[] screenData)
        {
            bool grayscale = t.Shape.Depth == 1;
            
            if (!grayscale && t.Shape.Depth != 3)
                throw new Exception("Invalid tensor depth. Expected depth 3 for RGB.");

            if (screenData.Length != t.Shape.Length)
                throw new Exception($"Invalid tensor length. Expected room for {screenData.Length} values.");

            for (int i = 0; i < screenData.Length; ++i)
                t.SetFlat(screenData[i] / 255.0f, i);
        }
    }
}
