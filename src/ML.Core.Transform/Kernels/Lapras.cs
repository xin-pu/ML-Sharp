using System.Linq;
using FluentAssertions;
using Numpy;

namespace ML.Core.Transform
{
    public class Lapras : Kernel
    {
        private double _beta;

        /// <summary>
        ///     拉普拉斯核模型
        ///     h(i,j) =  e^(-beta*Norm1(xi-xj))
        /// </summary>
        /// <param name="beta"></param>
        public Lapras(double beta = 1)
            : base(KernelType.Lapras)
        {
            beta.Should().BeGreaterOrEqualTo(0, "Please give Beta != 0 ");
            Beta = beta;
        }

        public double Beta
        {
            get => _beta;
            set => Set(ref _beta, value);
        }

        public override NDarray Call(NDarray input)
        {
            input.ndim.Should().Be(2, "input dims shoulbe be 2");
            var batchSize = input.shape[0];

            var output = np.zeros(batchSize, batchSize);

            Enumerable.Range(0, batchSize)
                .ToList()
                .ForEach(i =>
                {
                    var delta = input[i] - input;
                    var res = np.linalg.norm(delta, 1, -1);
                    output[i] = (-Beta * res).exp();
                });

            return output;
        }
    }
}