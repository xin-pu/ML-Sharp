using AutoDiff;
using FluentAssertions;
using Numpy;

namespace ML.Utility
{
    public class TermOp
    {
        /// <summary>
        /// </summary>
        /// <param name="v"></param>
        /// <param name="xrow"></param>
        /// <returns></returns>
        public static Term MatmulRow(Variable[] v, NDarray xrow)
        {
            xrow.shape[0].Should().Be(v.Length);
            var row = xrow.GetData<double>();
            var allTerms = row.Zip(v, (x, w) => w * x);
            var finalTerm = TermBuilder.Sum(allTerms);
            return finalTerm;
        }

        /// <summary>
        ///     Get list y_pred - y_true
        /// </summary>
        /// <param name="v"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public static Term[] Matmul(Variable[] v, NDarray x)
        {
            x.shape[1].Should().Be(v.Length);
            var batchSize = x.shape[0];
            var batchY = Enumerable.Range(0, batchSize).Select(r => MatmulRow(v, x[r])).ToArray();
            return batchY;
        }

        /// <summary>
        ///     ND array Multiply Variable
        /// </summary>
        /// <param name="x"></param>
        /// <param name="variables"></param>
        /// <returns></returns>
        public static TermMatrix Multiply(NDarray x, Variable[] variables)
        {
            var features = x.shape[1];
            var variablsLength = variables.Length;
            (variablsLength % features).Should().Be(0);

            var batchSize = x.shape[0];
            var labels = variablsLength / features;
            var varDict = variables
                .Select((v, i) => (v, i / features))
                .GroupBy(p => p.Item2, p => p.v)
                .ToDictionary(p => p.Key, p => p.ToArray());
            var matrix = new TermMatrix(labels, batchSize);

            foreach (var b in Enumerable.Range(0, batchSize))
            foreach (var v in varDict)
                matrix[b, v.Key] = MatmulRow(v.Value, x[b]);

            return matrix;
        }

        /// <summary>
        ///     Sigmoid 函数 Logistic 函数
        /// </summary>
        /// <param name="x"></param>
        /// <param name="weight"></param>
        /// <returns></returns>
        public static Term Sigmoid(Term x, double weight = 1)
        {
            return 1.0 / (TermBuilder.Exp(-weight * x) + 1.0);
        }

        /// <summary>
        ///     Sigmoid 函数 Tanh 函数
        /// </summary>
        /// <param name="x"></param>
        /// <param name="weight"></param>
        /// <returns></returns>
        public static Term Tanh(Term x, double weight = 1)
        {
            return 2.0 / (TermBuilder.Exp(-weight * x) + 1.0) - 1;
        }
    }
}