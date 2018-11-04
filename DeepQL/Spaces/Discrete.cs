using Neuro.Tensors;

namespace DeepQL.Spaces
{
    public class Discrete : Space
    {
        public Discrete(int n) : base(new Shape(1))
        {
            N = n;
        }

        public override Tensor Sample()
        {
            return new Tensor(new double[] { GlobalRandom.Rng.Next(N) }, Shape);
        }

        public override bool Contains(Tensor state)
        {
            return state.Shape.Equals(Shape) && state[0] >= 0 && state[0] < N;
        }

        private readonly int N;
    }
}
