using System;
using Neuro.Tensors;

namespace DeepQL.Spaces
{
    public abstract class Space
    {
        protected Space(Shape shape)
        {
            Shape = shape;
        }

        public abstract Tensor Sample();
        public abstract bool Contains(Tensor state);

        public virtual int NumberOfValues()
        {
            throw new Exception("Number of values in this space is infinite.");
        }

        public readonly Shape Shape;
    }
}
