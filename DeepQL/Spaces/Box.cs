using System;
using Neuro.Tensors;

namespace DeepQL.Spaces
{
    public class Box : Space
    {
        public Box(int low, int high, Shape shape)
            : base(shape)
        {
            Low = low;
            High = high;
        }

        public override Tensor Sample()
        {
            var t = new Tensor(Shape);
            t.Map(x => GlobalRandom.Rng.Next(Low, High + 1), t);
            return t;
        }

        public override bool Contains(Tensor state)
        {
            for (int i = 0; i < Shape.Length; ++i)
            {
                if (state.GetFlat(i) < Low || state.GetFlat(i) > High)
                    return false;
            }

            return true;
        }

        public readonly int Low;
        public readonly int High;
    }
}
