using System.Collections.ObjectModel;
using ML.Core.Data;
using ML.Core.Losses;
using ML.Core.Metrics;
using ML.Core.Metrics.Regression;
using ML.Core.Models;
using ML.Core.Optimizers;
using ML.Core.Trainers;

namespace ML.Guide
{
    /// <summary>
    ///     Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow
    {
        public MainWindow()
        {
            InitializeComponent();
            Initial();
            DataContext = Data;
        }

        public GDTrainer<DataView> Data { set; get; }

        private void Initial()
        {
            Data = new GDTrainer<DataView>
            {
                TrainDataset = new Dataset<DataView>(new DataView[] { }),
                ValDataset = new Dataset<DataView>(new DataView[] { }),
                ModelGd = new BinaryLogicClassify<DataView>(),
                Optimizer = new Nesterov(1E-2),
                Loss = new BinaryCrossentropy(),

                TrainPlan = new TrainPlan {Epoch = 100, BatchSize = 25},
                Metrics = new ObservableCollection<Metric> {new MeanAbsoluteError()}
            };
        }
    }
}