using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AutoDiff;
using FluentAssertions;
using GalaSoft.MvvmLight;
using GalaSoft.MvvmLight.CommandWpf;
using Microsoft.Win32;
using ML.Core.Data;
using ML.Core.Losses;
using ML.Core.Metrics;
using ML.Core.Models;
using ML.Core.Optimizers;
using Numpy;

namespace ML.Core.Trainers
{
    public class GDTrainer : ViewModelBase
    {
        private Loss _loss;

        private ObservableCollection<Metric> _metrics;
        private IModelGD _modelGd;
        private Optimizer _optimizer;
        private Dataset<DataView> _trainDataset;

        private TrainPlan _trainPlan;
        private Dataset<DataView> _valDataset;

        public Action<GDTrainer> AfterBatchPipeline;
        public Action<GDTrainer> AfterEpochPipeline;

        public Action<GDTrainer> BeforeBatchPipeline;
        public Action<GDTrainer> BeforeEpochPipeline;

        public Action<string> Print;


        public IModelGD ModelGd
        {
            get => _modelGd;
            set => Set(ref _modelGd, value);
        }

        public Dataset<DataView> TrainDataset
        {
            get => _trainDataset;
            set => Set(ref _trainDataset, value);
        }

        public Dataset<DataView> ValDataset
        {
            get => _valDataset;
            set => Set(ref _valDataset, value);
        }

        public Optimizer Optimizer
        {
            get => _optimizer;
            set => Set(ref _optimizer, value);
        }

        public Loss Loss
        {
            get => _loss;
            set => Set(ref _loss, value);
        }

        public ObservableCollection<Metric> Metrics
        {
            get => _metrics;
            set => Set(ref _metrics, value);
        }

        public TrainPlan TrainPlan
        {
            get => _trainPlan;
            set => Set(ref _trainPlan, value);
        }


        public async Task Fit()
        {
            TrainDataset.Should().NotBeNull("dataset should not ne null");

            ModelGd.PipelineDataSet(TrainDataset);

            foreach (var e in Enumerable.Range(0, TrainPlan.Epoch))
                await Task.Run(() =>
                {
                    BeforeEpochPipeline?.Invoke(this);

                    var iEnumerator = TrainDataset.GetEnumerator(TrainPlan.BatchSize);

                    while (iEnumerator.MoveNext() &&
                           iEnumerator.Current is Dataset<DataView> data)
                    {
                        if (data.Count == 0)
                            continue;

                        BeforeBatchPipeline?.Invoke(this);
                        var batchdataSet = data.ToDatasetNDarray();

                        var predTerms = ModelGd.CallGraph(batchdataSet.Feature);
                        var lossTerm = Loss.GetLossTerm(predTerms, batchdataSet.Label, ModelGd.Variables);


                        NDarray GetGradient(NDarray weight)
                        {
                            var gradientArray = lossTerm.Differentiate(ModelGd.Variables, weight.GetData<double>());
                            var g = np.array(gradientArray);
                            return np.reshape(g, weight.shape);
                        }

                        ModelGd.Weights = Optimizer.Call(ModelGd.Weights, GetGradient, e);

                        AfterBatchPipeline?.Invoke(this);
                    }

                    var trainMsg = new StringBuilder($"#{e + 1:D4}\t");
                    var train_loss = UpdateLossMetric(TrainDataset);
                    trainMsg.Append($"[Loss]:{train_loss:F4}\t");
                    foreach (var metric in Metrics) trainMsg.Append($"{metric}\t");

                    if (ValDataset != null)
                    {
                        var val_loss = UpdateLossMetric(ValDataset);
                        trainMsg.Append("\tVal\t");
                        trainMsg.Append($"[Loss]:{val_loss:F4}\t");
                        foreach (var metric in Metrics) trainMsg.Append($"{metric}\t");
                    }

                    Print?.Invoke(trainMsg.ToString());

                    AfterEpochPipeline?.Invoke(this);
                });

            /// early stoping
            /// Print status of each epoch
        }

        public double UpdateLossMetric(Dataset<DataView> dataset)
        {
            var dataview = dataset.ToDatasetNDarray();
            var y_pred = ModelGd.Call(dataview.Feature);
            var y_true = dataview.Label;
            var predterms = ModelGd.CallGraph(dataview.Feature);
            var lossTerm = Loss.GetLossTerm(predterms, dataview.Label, ModelGd.Variables);
            var loss = lossTerm.Evaluate(ModelGd.Variables, ModelGd.GetWeightArray());
            Metrics.ToList().ForEach(m => m.Call(y_pred, y_true));

            return loss;
        }

        #region DataSet Command

        #endregion

        #region Model Command

        public RelayCommand<Type> ChangeModelCommand => new(modelType => ChangeModelTypeCommand_Execute(modelType));

        private void ChangeModelTypeCommand_Execute(Type modelType)
        {
            ModelGd = Activator.CreateInstance(modelType) as ModelGD;
        }

        public RelayCommand SaveModelCommand => new(() => SaveModelCommand_Execute());

        private void SaveModelCommand_Execute()
        {
            var saveDialog = new SaveFileDialog
            {
                Filter = @"XML(*.xml)|*.xml"
            };
            var res = saveDialog.ShowDialog();
            if (res != true || saveDialog.FileName == "") return;

            ModelGd.Save(saveDialog.FileName);
        }

        public RelayCommand LoadModelCommand => new(() => LoadModelCommand_Execute());

        private void LoadModelCommand_Execute()
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = @"aries(*.ar)|*.ar"
            };
            var res = openFileDialog.ShowDialog();
            if (res != true || openFileDialog.FileName == "")
                return;

            ModelGd = ModelGD.Load(openFileDialog.FileName);
        }

        #endregion

        #region Loss Command

        public RelayCommand<Type> ChangeLossCommand => new(lossType => ChangeLossCommand_Execute(lossType));


        private void ChangeLossCommand_Execute(Type lossType)
        {
            Loss = Activator.CreateInstance(lossType) as Loss;
        }

        #endregion

        #region Optimizer Command

        public RelayCommand<Type> ChangeOptimizerCommand =>
            new(optimizerType => ChangeOptimizerCommand_Execute(optimizerType));


        private void ChangeOptimizerCommand_Execute(Type optimizerType)
        {
            Optimizer = Activator.CreateInstance(optimizerType) as Optimizer;
        }

        #endregion

        #region Metric Command

        public RelayCommand<Metric> RemoveMetricCommand => new(metric => RemoveMetricCommand_Execute(metric));

        public RelayCommand<Type> AddMetricCommand => new(metricTYpe => AddMetricCommand_Execute(metricTYpe));

        public RelayCommand ClearMetricCommand => new(ClearMetricCommand_Execute);

        private void ClearMetricCommand_Execute()
        {
            Metrics.Clear();
        }

        private void RemoveMetricCommand_Execute(Metric metric)
        {
            if (Metrics.Contains(metric)) Metrics.Remove(metric);
        }

        private void AddMetricCommand_Execute(Type metricType)
        {
            var metric = Activator.CreateInstance(metricType) as Metric;
            Metrics.Add(metric);
        }

        #endregion

        #region Control Command

        #endregion


        #region Plan Save Load Command

        #endregion
    }
}