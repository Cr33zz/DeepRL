using System;
using System.Collections.Generic;
using System.Linq;

namespace DeepQL.Misc
{
    public class MovingAverage
    {
        public MovingAverage(int n)
        {
            N = n;
        }

        public void Add(float value)
        {
            if (Values.Count == N)
                Values.RemoveLast();

            Values.AddFirst(value);
        }

        public float AvgN(int n)
        {
            if (Values.Count == 0)
                return 0;

            float sum = 0;
            int i = 0;
            foreach (var v in Values)
            {
                sum += v;
                ++i;

                if (i == n)
                    break;
            }
            return sum / i;
        }

        public void Reset()
        {
            Values.Clear();
        }

        public readonly int N;
        public float Avg => Values.Sum() / Values.Count;

        private readonly LinkedList<float> Values = new LinkedList<float>();
    }
}
