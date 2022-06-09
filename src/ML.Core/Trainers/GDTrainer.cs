using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AutoDiff;
using FluentAssertions;
using GalaSoft.MvvmLight;
using ML.Core.Data;
using ML.Core.Losses;
using ML.Core.Metrics;
using ML.Core.Metrics.Regression;
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


        public Action<string> Print;

        public GDTrainer()
        {
            ModelGd = new MultipleLinearRegression();
            Optimizer = new Adam();
            Loss = new MeanSquared();
            Metrics = new ObservableCollection<Metric> {new MeanSquaredError(), new RSquared()};
            TrainPlan = new TrainPlan();
        }


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

        public async Task Fit(CancellationTokenSource cancellation = null)
        {
            await Task.Run(() =>
            {
                TrainDataset.Should().NotBeNull("dataset should not ne null");

                ModelGd.PipelineDataSet(TrainDataset);

                foreach (var e in Enumerable.Range(0, TrainPlan.Epoch))
                {
                    if (cancellation?.IsCancellationRequested == true)
                        return;


                    var iEnumerator = TrainDataset.GetEnumerator(TrainPlan.BatchSize);

                    while (iEnumerator.MoveNext() &&
                           iEnumerator.Current is Dataset<DataView> data)
                    {
                        if (cancellation?.IsCancellationRequested == true)
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

        private double UpdateLossMetric(Dataset<DataView> dataset)
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
    }
}