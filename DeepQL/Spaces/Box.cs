using System;
using Neuro.Tensors;

namespace DeepQL.Spaces
{
    public class Box : Space
    {
        public Box(double low, double high, Shape shape)
            : base(shape)
        {
            Low = new Tensor(shape);
            Low.FillWithValue(low);
            High = new Tensor(shape);
            High.FillWithValue(high);
        }

        public Box(double[] low, double[] high, Shape shape)
            : base(shape)
        {
            Low = new Tensor(low, shape);
            High = new Tensor(high, shape);
        }

        public override Tensor Sample()
        {
            var t = new Tensor(Shape);
            for (int i = 0; i < Shape.Length; ++i)
                t.SetFlat(Low.GetFlat(i) + (High.GetFlat(i) - Low.GetFlat(i)) * GlobalRandom.Rng.NextDouble(), i);
            return t;
        }

        public override bool Contains(Tensor state)
        {
            for (int i = 0; i < Shape.Length; ++i)
            {
                if (state.GetFlat(i) < Low.GetFlat(i) || state.GetFlat(i) > High.GetFlat(i))
                    return false;
            }

            return true;
        }

        public readonly Tensor Low;
        public readonly Tensor High;
    }
}
