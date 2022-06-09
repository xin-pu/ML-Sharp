using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AutoDiff;
using FluentAssertions;
using GalaSoft.MvvmLight;
using GalaSoft.MvvmLight.CommandWpf;
using Microsoft.Win32;
using ML.Core.Data;
using ML.Core.Data.Loader;
using ML.Core.Losses;
using ML.Core.Metrics;
using ML.Core.Models;
using ML.Core.Optimizers;
using Numpy;

namespace ML.Core.Trainers
{
    public class GDTrainer : ViewModelBase
    {
        private Loss _loss = new MeanSquared();
        private ObservableCollection<Metric> _metrics = new();
        private IModelGD _modelGd = new MultipleLinearRegression();
        private Optimizer _optimizer = new SGD();
        private Dataset<DataView> _trainDataset;
        private TrainPlan _trainPlan = new();
        private Dataset<DataView> _valDataset;


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


        #region DataSet Command

        public RelayCommand<Type> LoadTrainDatasetCommand => new(datatype => LoadTrainDatasetCommand_Execute(datatype));

        public RelayCommand<Type> LoadValDatasetCommand => new(datatype => LoadValDatasetCommand_Execute(datatype));

        private void LoadTrainDatasetCommand_Execute(Type datatype)
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = @"txt(*.txt)|*.txt"
            };
            var res = openFileDialog.ShowDialog();
            if (res != true || openFileDialog.FileName == "")
                return;
            TrainDataset = TextLoader.LoadDataSet(openFileDialog.FileName, datatype, false);
        }

        private void LoadValDatasetCommand_Execute(Type datatype)
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = @"txt(*.txt)|*.txt"
            };
            var res = openFileDialog.ShowDialog();
            if (res != true || openFileDialog.FileName == "")
                return;
            ValDataset = TextLoader.LoadDataSet(openFileDialog.FileName, datatype, false);
        }

        #endregion


        #region Control Command

        public RelayCommand TrainCommand => new(() => TrainCommand_Execute());
        public RelayCommand CancelCommand => new(() => CancelCommand_Execute());
        public CancellationTokenSource CancellationTokenSource { internal set; get; } = new();

        private async void TrainCommand_Execute()
        {
            CancellationTokenSource = new CancellationTokenSource();

            try
            {
                PreCheck();
                await Fit();
            }
            catch (Exception ex)
            {
                // ignored
            }
        }


        public void PreCheck()
        {
        }

        public async Task Fit()
        {
            await Task.Run(() =>
            {
                TrainDataset.Should().NotBeNull("dataset should not ne null");

                ModelGd.PipelineDataSet(TrainDataset);

                foreach (var e in Enumerable.Range(0, TrainPlan.Epoch))
                {
                    if (CancellationTokenSource.IsCancellationRequested)
                        return;


                    var iEnumerator = TrainDataset.GetEnumerator(TrainPlan.BatchSize);

                    while (iEnumerator.MoveNext() &&
                           iEnumerator.Current is Dataset<DataView> data)
                    {
                        if (CancellationTokenSource.IsCancellationRequested)
                            return;

                        if (data.Count == 0)
                            continue;

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
                    }

                    var trainMsg = new StringBuilder($"#{e + 1:D4}\t");
                    var train_loss = UpdateLossMetric(TrainDataset);
                    trainMsg.Append($"[Loss]:{train_loss:F4}\t");

                    if (ValDataset != null && (ValDataset != null || ValDataset.Value != null))
                    {
                        var val_loss = UpdateLossMetric(ValDataset);
                        trainMsg.Append("\tVal\t");
                        trainMsg.Append($"[Loss]:{val_loss:F4}\t");
                        foreach (var metric in Metrics) trainMsg.Append($"{metric}\t");
                    }

                    Print?.Invoke(trainMsg.ToString());
                }

                /// early stoping
                /// Print status of each epoch
            });
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

        private void CancelCommand_Execute()
        {
            CancellationTokenSource.Cancel();
        }

        #endregion
    }
}