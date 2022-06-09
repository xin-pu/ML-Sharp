using System;
using System.Linq;
using GalaSoft.MvvmLight;
using GalaSoft.MvvmLight.CommandWpf;
using ML.Core.Losses;
using ML.Core.Metrics;
using ML.Core.Models;
using ML.Core.Optimizers;
using ML.Core.Trainers;

namespace ML.Guide.ViewModel.Menu
{
    public class MenuOption : ViewModelBase
    {
        public GDTrainer GDTrainner => ViewModelLocator.Instance.GDTrainner;

        private void ChangeLossCommand_Execute(Type lossType)
        {
            GDTrainner.Loss = Activator.CreateInstance(lossType) as Loss;
        }

        private void ChangeOptimizerCommand_Execute(Type optimizerType)
        {
            GDTrainner.Optimizer = Activator.CreateInstance(optimizerType) as Optimizer;
        }

        private void ClearMetricCommand_Execute()
        {
            GDTrainner.Metrics.Clear();
        }

        private void RemoveMetricCommand_Execute(Metric metric)
        {
            var Metrics = GDTrainner.Metrics;
            if (Metrics.Contains(metric)) Metrics.Remove(metric);
        }

        private void AddMetricCommand_Execute(Type metricType)
        {
            var Metrics = GDTrainner.Metrics;
            if (Metrics.Select(a => a.GetType()).Contains(metricType))
                return;
            var metric = Activator.CreateInstance(metricType) as Metric;
            Metrics.Add(metric);
        }

        private void ChangeModelTypeCommand_Execute(Type modelType)
        {
            GDTrainner.ModelGd = Activator.CreateInstance(modelType) as ModelGD;
        }


        #region Command

        /// <summary>
        ///     修改模型
        /// </summary>
        public RelayCommand<Type> ChangeModelCommand =>
            new(modelType => ChangeModelTypeCommand_Execute(modelType));

        /// <summary>
        ///     修改优化器
        /// </summary>
        public RelayCommand<Type> ChangeOptimizerCommand =>
            new(optimizerType => ChangeOptimizerCommand_Execute(optimizerType));


        /// <summary>
        ///     修改损失
        /// </summary>
        public RelayCommand<Type> ChangeLossCommand =>
            new(lossType => ChangeLossCommand_Execute(lossType));

        /// <summary>
        ///     移除评估器
        /// </summary>
        public RelayCommand<Metric> RemoveMetricCommand =>
            new(metric => RemoveMetricCommand_Execute(metric));

        /// <summary>
        ///     添加评估器
        /// </summary>
        public RelayCommand<Type> AddMetricCommand =>
            new(metricTYpe => AddMetricCommand_Execute(metricTYpe));

        /// <summary>
        ///     清除评估器
        /// </summary>
        public RelayCommand ClearMetricCommand =>
            new(ClearMetricCommand_Execute);

        #endregion
    }
}