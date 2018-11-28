using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepQL
{
    public static partial class RandomExtensions
    {
        public static float NextFloat(this Random rng, float low, float high)
        {
            return low + (high - low) * (float)rng.NextDouble();
        }

        public static int[] Next(this Random rng, int low, int high, int size)
        {
            var result = new int[size];
            for (int i = 0; i < size; ++i)
                result[i] = rng.Next(low, high);
            return result;
        }

        public static float[] Nextfloat(this Random rng, float low, float high, int size)
        {
            var result = new float[size];
            for (int i = 0; i < size; ++i)
                result[i] = rng.NextFloat(low, high);
            return result;
        }

        public static float[] NextFloat(this Random rng, float low, float high, int size)
        {
            var result = new float[size];
            for (int i = 0; i < size; ++i)
                result[i] = (float)rng.NextFloat(low, high);
            return result;
        }
    }
}
