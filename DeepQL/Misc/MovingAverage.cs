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

        public void Add(double value)
        {
            if (Values.Count == N)
                Values.Dequeue();

            Values.Enqueue(value);
        }

        public double Avg => Values.Sum() / Values.Count;

        private Queue<double> Values = new Queue<double>();
        private readonly int N;
    }
}
