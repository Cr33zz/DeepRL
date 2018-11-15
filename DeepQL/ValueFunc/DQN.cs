using System.Collections.Generic;
using Neuro;
using Neuro.Layers;
using Neuro.Optimizers;
using Neuro.Tensors;

namespace DeepQL.ValueFunc
{
    public class DQN : ValueFunctionModel
    {
        public DQN(Shape inputShape, int numberOfActions, double learningRate, double discountFactor, int replaySize = 2000, int batchSize = 32)
            : base(inputShape, numberOfActions, learningRate, discountFactor)
        {
            Model = new NeuralNetwork("DQN_agent");
            Model.AddLayer(new Flatten(inputShape));
            Model.AddLayer(new Dense(Model.LastLayer(), 24, Activation.ReLU));
            Model.AddLayer(new Dense(Model.LastLayer(), 24, Activation.ReLU));
            Model.AddLayer(new Dense(Model.LastLayer(), numberOfActions, Activation.Linear));
            Model.Optimize(new Adam(learningRate), Loss.MeanSquareError);

            ReplayMem = new ReplayMemory(replaySize);
            BatchSize = batchSize;
        }

        public override Tensor GetOptimalAction(Tensor state)
        {
            var qValues = Model.Predict(state);
            var action = new Tensor(new Shape(1));
            action[0] = qValues.ArgMax();
            return action;
        }

        public override void OnTransition(Tensor state, Tensor action, double reward, Tensor nextState, bool done)
        {
            ReplayMem.Push(new Transition(state, action, reward, nextState, done));

            if (ReplayMem.StorageSize >= BatchSize)
                Train(ReplayMem.Sample(BatchSize));
        }

        public override void SaveState(string filename)
        {
            Model.SaveStateXml(filename);
        }

        public override void LoadState(string filename)
        {
            Model.LoadStateXml(filename);
        }

        protected override void Train(List<Transition> transitions)
        {
            foreach (var trans in transitions)
            {
                // calculate new predicted reward
                var target = trans.Reward;
                if (!trans.Done)
                    target = trans.Reward + DiscountFactor * Model.Predict(trans.NextState).Max();

                var target_f = Model.Predict(trans.State); // this is our original prediction
                target_f[(int)trans.Action[0]] = target; // this is the expected prediction for selected action

                Model.Fit(trans.State, target_f, 1, 0, Track.Nothing);
            }
        }

        protected NeuralNetwork Model;
        private ReplayMemory ReplayMem;
        private int BatchSize;
    }
}
