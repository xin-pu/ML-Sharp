using System;
using System.Text.RegularExpressions;
using FluentAssertions;
using GalaSoft.MvvmLight;
using Numpy;

namespace ML.Core.Metrics
{
    public abstract class Metric : ViewModelBase, IRecorder
    {
        private string _logogram;
        private string _name;
        private double _valueError;

        protected Metric()
        {
            Name = GetType().Name;
            Logogram = Regex.Replace(Name, @"[^A-Z]+", "");
        }

        /// <summary>
        ///     误差
        /// </summary>
        public double ValueError
        {
            protected set => Set(ref _valueError, value);
            get => _valueError;
        }

        /// <summary>
        ///     简写名
        /// </summary>
        public string Logogram
        {
            protected set => Set(ref _logogram, value);
            get => _logogram;
        }

        /// <summary>
        ///     描述
        /// </summary>
        public abstract string Describe { get; }

        /// <summary>
        ///     指标名
        /// </summary>
        public string Name
        {
            protected set => Set(ref _name, value);
            get => _name;
        }

        public Action<double> ReportToRecorder { get; set; }


        internal virtual void precheck(NDarray y_true, NDarray y_pred)
        {
            y_true.shape.Should().BeEquivalentTo(y_pred.shape, "shape of y_true and y_pred should be same.");
        }

        internal abstract double call(NDarray y_true, NDarray y_pred);

        public double Call(NDarray y_true, NDarray y_pred)
        {
            ValueError = call(y_true, y_pred);
            ReportToRecorder?.Invoke(ValueError);
            return ValueError;
        }

        public override string ToString()
        {
            return $"[{Logogram}]:{ValueError:F4}";
        }
    }
}