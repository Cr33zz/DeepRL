using Neuro.Tensors;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepQL
{
    public abstract class Environment
    {
        public abstract double Step(int action, Tensor result);
        public abstract Shape StateShape();
    }
}
