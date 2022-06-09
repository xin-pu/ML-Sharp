using System;
using System.Linq;
using GalaSoft.MvvmLight;
using GalaSoft.MvvmLight.CommandWpf;
using ML.Core.Losses;
using ML.Core.Metrics;
using ML.Core.Models;
using ML.Core.Optimizers;

namespace ML.Guide.ViewModel.Menu
{
    public class MenuOption : ViewModelBase
    {
        public RelayCommand<Type> ChangeOptimizerCommand =>
            new(optimizerType => ChangeOptimizerCommand_Execute(optimizerType));


        public RelayCommand<Type> ChangeLossCommand =>
            new(lossType => ChangeLossCommand_Execute(lossType));


        public RelayCommand<Metric> RemoveMetricCommand => new(metric => RemoveMetricCommand_Execute(metric));

        public RelayCommand<Type> AddMetricCommand => new(metricTYpe => AddMetricCommand_Execute(metricTYpe));

        public RelayCommand ClearMetricCommand => new(ClearMetricCommand_Execute);


        public RelayCommand<Type> ChangeModelCommand => new(modelType => ChangeModelTypeCommand_Execute(modelType));


        private void ChangeLossCommand_Execute(Type lossType)
        {
            ViewModelLocator.Instance.GDTrainner.Loss = Activator.CreateInstance(lossType) as Loss;
        }

        private void ChangeOptimizerCommand_Execute(Type optimizerType)
        {
            ViewModelLocator.Instance.GDTrainner.Optimizer = Activator.CreateInstance(optimizerType) as Optimizer;
        }

        private void ClearMetricCommand_Execute()
        {
            ViewModelLocator.Instance.GDTrainner.Metrics.Clear();
        }

        private void RemoveMetricCommand_Execute(Metric metric)
        {
            var Metrics = ViewModelLocator.Instance.GDTrainner.Metrics;
            if (Metrics.Contains(metric)) Metrics.Remove(metric);
        }

        private void AddMetricCommand_Execute(Type metricType)
        {
            var Metrics = ViewModelLocator.Instance.GDTrainner.Metrics;
            if (Metrics.Select(a => a.GetType()).Contains(metricType))
                return;
            var metric = Activator.CreateInstance(metricType) as Metric;
            Metrics.Add(metric);
        }

        private void ChangeModelTypeCommand_Execute(Type modelType)
        {
            ViewModelLocator.Instance.GDTrainner.ModelGd = Activator.CreateInstance(modelType) as ModelGD;
        }
    }
}