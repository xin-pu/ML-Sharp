using System;
using System.Linq;
using ML.Core.Transform;
using Numpy;

namespace MLNet.Transforms
{
    public class Poly : Kernel
    {
        public Poly(int degree = 2)
            : base(KernelType.Poly)
        {
            if (degree <= 0)
                throw new ArgumentException("Please give degree > 0 ");
            Degree = degree;
        }

        public int Degree { protected set; get; }

        public override NDarray Call(NDarray input)
        {
            var p = input.shape[0];

            var output = np.zeros(p, p);


            var x_array = Enumerable.Range(0, p)
                .Select(r => input[$"{r},:"]).ToList();

            Enumerable.Range(0, p)
                .AsParallel()
                .ToList()
                .ForEach(i => { output[i] = np.power(1 + np.dot(input, x_array[i]), np.array(Degree)) - 1; });
            return output;
        }
    }
}