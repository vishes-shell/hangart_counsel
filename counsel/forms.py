from django.forms import ModelForm

from .models import RatingPerson


class RatingPersonEmailForm(ModelForm):
    """
        Description: The asking for promo after counsel form.
    """

    class Meta:
        model = RatingPerson
        fields = ('email',)

    def __init__(self, *args, **kwargs):
        super(RatingPersonEmailForm, self).__init__(*args, **kwargs)

        self.fields['email'].widget.attrs['type'] = 'email'
        self.fields['email'].required = True
        self.fields['email'].widget.attrs['name'] = 'email'
        self.fields['email'].widget.attrs['placeholder'] = 'Адрес электронной почты'
        self.fields['email'].widget.attrs['id'] = 'email'
        self.fields['email'].widget.attrs['value'] = ''
        self.fields['email'].error_messages = {'required': 'Введите Ваш адрес электронной почты.'}
        self.fields['email'].label = ''
