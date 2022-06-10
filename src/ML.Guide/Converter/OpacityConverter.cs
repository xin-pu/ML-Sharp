using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace ML.Guide.Converter
{
    public class OpacityConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value != null && (Visibility) value == Visibility.Visible ? 1d : .2d;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}