using System;
using System.Collections.Generic;
using System.Linq;

namespace DeepQL.MemoryReplays
{
    public class PriorityExperienceReplay : BaseExperienceReplay
    {
        public PriorityExperienceReplay(int capacity, float alpha = 0.6f, float beta = 0.4f)
            : base(capacity)
        {
            Tree = new SumTree(capacity);
            Alpha = alpha;
            Beta = beta;
        }

        public override void Push(Experience trans)
        {
            if (trans == null)
                throw new ArgumentNullException();

            float maxPriority = Tree.GetMaxPriority(Size);

            if (maxPriority == 0)
                maxPriority = MaxError;

            Tree.Add(trans, maxPriority);

            if (Size < Capacity)
                ++Size;
        }

        public override List<Experience> Sample(int batchSize)
        {
            var sample = new List<Experience>();

            float prioritySegment = Tree.GetTotalPriority() / batchSize;

            Beta += BetaIncrement;
            if (Beta > 1)
                Beta = 1;

            float pMin = Tree.GetMinPriority(Size) / Tree.GetTotalPriority();
            float maxWeight = (float)Math.Pow(pMin * batchSize, -Beta);

            for (int i = 0; i < batchSize; ++i)
            {
                // sample value uniformly from each priority segment range
                float value = GlobalRandom.Rng.NextFloat(prioritySegment * i, prioritySegment * (i + 1));

                var exp = Tree.GetLeaf(value, out var priority);

                float samplingProb = priority / Tree.GetTotalPriority();
                exp.ImportanceSamplingWeight = (float)Math.Pow(batchSize * samplingProb, -Beta) / maxWeight;

                sample.Add(exp);
            }

            return sample;
        }

        public override void Update(List<Experience> samples, List<float> absErrors)
        {
            for (int i = 0; i < samples.Count; ++i)
            {
                float clippedAbsError = Math.Min(absErrors[i] + Epsilon, MaxError);
                float priority = (float) Math.Pow(clippedAbsError, Alpha);
                Tree.Update(samples[i].Index, priority);
            }
        }

        public override int GetSize()
        {
            return Size;
        }

        public override string GetParametersDescription()
        {
            return $"{base.GetParametersDescription()} α={Alpha} β={Beta} β_increment={BetaIncrement} ε={Epsilon} max_error={MaxError}";
        }

        private readonly SumTree Tree;
        
        // Hyper-parameter used to make a trade off between taking only exp with high priority and sampling randomly (uniform selection when 0)
        private readonly float Alpha;
        // Importance-sampling, it will be annealing from initial value to 1 every time we sample from memory
        private float Beta;
        private readonly float BetaIncrement = 0.001f;
        private readonly float Epsilon = 0.01f;
        private readonly float MaxError = 1;
        private int Size;
    }    
}
