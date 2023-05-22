using System.Text;
using AutoDiff;
using CommunityToolkit.Mvvm.ComponentModel;
using ML.Core.Data;
using ML.Utility;
using Numpy;

namespace ML.Core.Models.NeuralNets
{
    /// <summary>
    ///     用于梯度下降法的基本模型
    ///     需要定义损失函数
    ///     需要定义优化器
    ///     需要定义转换器
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class Module : ObservableObject
    {
        private bool _isWeightInitialed;
        private WeightInitial _weightInitial = WeightInitial.Rand;
        private Variable[] _variables = Array.Empty<Variable>();
        private NDarray _weights = np.empty();

        /// <summary>
        /// </summary>
        protected Module(Dtype dtype)
        {
            Dtype = dtype;
        }

        public Dtype Dtype { internal set; get; }

        public abstract string Description { get; }

        public string Name => GetType().Name;

        public WeightInitial WeightInitialMode
        {
            protected set => SetProperty(ref _weightInitial, value);
            get => _weightInitial;
        }

        public bool IsWeightInitialed
        {
            set => SetProperty(ref _isWeightInitialed, value);
            get => _isWeightInitialed;
        }


        public abstract void PipeLineForWeights(NDarray input);


        /// <summary>
        /// </summary>
        /// <param name="features">[batch size, d1, d2, ... ]</param>
        /// <returns></returns>
        public abstract TermMatrix CallGraph(NDarray features);

        public abstract NDarray Call(NDarray features);

        public NDarray Call(DataView data)
        {
            var datas = new Dataset<DataView>(new[] {data});
            return Call(datas.ToDatasetNDarray().Feature)[0];
        }


        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine(Name);
            str.AppendLine(Description);

            return str.ToString();
        }
    }
}