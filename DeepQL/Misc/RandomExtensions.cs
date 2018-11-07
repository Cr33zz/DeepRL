using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepQL
{
    public static partial class RandomExtensions
    {
        public static double NextDouble(this Random rng, double low, double high)
        {
            return low + (high - low) * rng.NextDouble();
        }
    }
}
