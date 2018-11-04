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

        public readonly Shape Shape;
    }
}
