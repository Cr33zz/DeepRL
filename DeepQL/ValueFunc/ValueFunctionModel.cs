using System.Collections.Generic;
using Neuro.Tensors;

namespace DeepQL.ValueFunc
{
    public abstract class ValueFunctionModel
    {
        protected ValueFunctionModel(Shape inputShape, int numberOfActions, float learningRate, float discountFactor)
        {
            LearningRate = learningRate;
            DiscountFactor = discountFactor;
            NumberOfActions = numberOfActions;
        }

        public abstract Tensor GetOptimalAction(Tensor state);

        public abstract void OnStep(Tensor state, Tensor action, float reward, Tensor nextState, bool done);
        public abstract void OnTrain();
        public virtual void OnEpisodeEnd(int episode) { }
        public virtual void SaveState(string filename) { }
        public virtual void LoadState(string filename) { }
        public virtual string GetParametersDescription() { return $"lr={LearningRate} γ={DiscountFactor}"; }

        protected readonly float LearningRate;
        protected readonly float DiscountFactor;
        public readonly int NumberOfActions;
    }
}
