using System.Windows;
using System.Windows.Controls;

namespace ML.Guide.Controls.Basic
{
    public class TopTabButton : Button
    {
        public static readonly DependencyProperty IconProperty =
            DependencyProperty.Register("Icon", typeof(string), typeof(TopTabButton), new PropertyMetadata(null));

        static TopTabButton()
        {
            DefaultStyleKeyProperty.OverrideMetadata(typeof(TopTabButton),
                new FrameworkPropertyMetadata(typeof(TopTabButton)));
        }

        public string Icon
        {
            get => (string) GetValue(IconProperty);
            set => SetValue(IconProperty, value);
        }
    }
}