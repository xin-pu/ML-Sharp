using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using LiveCharts.Wpf;

namespace ML.Guide.Views
{
    /// <summary>
    ///     Interaction logic for RecordView.xaml
    /// </summary>
    public partial class RecordView
    {
        public RecordView()
        {
            InitializeComponent();
        }

        private void ListBox_OnPreviewMouseDown(object sender, MouseButtonEventArgs e)
        {
            var item = ItemsControl.ContainerFromElement(ListBox, (DependencyObject) e.OriginalSource)
                as ListBoxItem;
            if (item == null) return;

            var series = (StackedAreaSeries) item.Content;
            series.Visibility = series.Visibility == Visibility.Visible
                ? Visibility.Hidden
                : Visibility.Visible;
        }
    }
}