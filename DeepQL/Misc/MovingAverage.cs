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
                Values.Dequeue();

            Values.Enqueue(value);
        }

        public readonly int N;
        public float Avg => Values.Sum() / Values.Count;

        private readonly Queue<float> Values = new Queue<float>();
    }
}
