using System.Linq;
using System.Text;
using AutoDiff;
using FluentAssertions;
using ML.Core.Data;
using ML.Core.Transform;
using MvvmCross.ViewModels;
using Numpy;

namespace ML.Core.Models
{
    /// <summary>
    ///     用于梯度下降法的基本模型
    ///     需要定义损失函数
    ///     需要定义优化器
    ///     需要定义转换器
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class Model<T> : MvxViewModel
        where T : DataView
    {
        private InitialWeigts _initialWeights = InitialWeigts.False;
        private Variable[] _variables;

        private WeightInitial _weightInitial;
        private NDarray _weights;

        public string Name => GetType().Name;

        public abstract Transformer Transformer { set; get; }

        public Variable[] Variables
        {
            get => _variables;
            protected set => SetProperty(ref _variables, value);
        }

        public NDarray Weights
        {
            get => _weights;
            protected set => SetProperty(ref _weights, value);
        }

        public double[] WeightsArray => Weights?.GetData<double>();

        public InitialWeigts InitialWeights
        {
            get => _initialWeights;
            protected set => SetProperty(ref _initialWeights, value);
        }

        public WeightInitial WeightInitial
        {
            get => _weightInitial;
            protected set => SetProperty(ref _weightInitial, value);
        }

        /// <summary>
        ///     1.通过传入数据集=>模型变换，确认参数数量
        /// </summary>
        /// <param name="dataset"></param>
        public void PipelineDataSet(Dataset<T> dataset)
        {
            var dataview = dataset.ToDatasetNDarray();
            var transformNDarray = Transformer.Call(dataview.Feature);

            var featureCount = transformNDarray.shape[1];
            var labelCount = dataview.Label.shape[1];
            var enumerable = Enumerable.Range(0, featureCount);

            Variables = enumerable
                .Select(i => new Variable())
                .ToArray();


            switch (WeightInitial)
            {
                case WeightInitial.One:
                    Weights = np.ones(featureCount, labelCount);
                    break;
                case WeightInitial.Zero:
                    Weights = np.zeros(featureCount, labelCount);
                    break;
                case WeightInitial.Rand:
                default:
                    Weights = np.random.rand(featureCount, labelCount);
                    break;
            }

            InitialWeights = InitialWeigts.True;
        }

        public void UpdateWeights(NDarray weightNDarray)
        {
            weightNDarray.shape.Should().BeEquivalentTo(Weights.shape, "Weigts shape should keep.");

            Weights = weightNDarray;
        }

        /// <summary>
        /// </summary>
        /// <param name="x">[batch size, d1, d2, ... ]</param>
        /// <returns></returns>
        public abstract Term[] CallGraph(NDarray x);

        /// <summary>
        ///     call by X*W
        /// </summary>
        /// <param name="x">[batch size, ... ]</param>
        /// <returns>[batch size, labels]</returns>
        public abstract NDarray Call(NDarray x);

        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine(Name);
            str.AppendLine($"{Transformer}");
            if (InitialWeights == InitialWeigts.True)
            {
                str.AppendLine($"ParaCount:\t{Variables.Length}");
                str.AppendLine($"ParaData:\r{Weights}");
            }

            return str.ToString();
        }
    }

    public enum WeightInitial
    {
        Rand,
        One,
        Zero
    }

    public enum InitialWeigts
    {
        True,
        False
    }
}