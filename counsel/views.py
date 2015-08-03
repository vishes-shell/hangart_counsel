from random import shuffle
import json

from django.http import Http404, HttpResponse
from django.shortcuts import get_object_or_404, render_to_response, RequestContext
from django.db.models import Count
from django.core.context_processors import csrf

from .models import Rating, RatingPerson
from .forms import RatingPersonEmailForm
from ..shop.models import Picture
from ..users.forms import UserForm
from ..shop.views import check_cart


def rating_change_session_user(session_key, user):
    """
        Changing user as instance with session key, to instance with user_id
    :param session_key:
    :param user:
    :return:
    """
    rating_people = RatingPerson.objects.filter(session_key=session_key)
    for person in rating_people:
        person.session_key = ''
        person.user_id = user
        person.save()
    ratings = Rating.objects.filter(session_key=session_key)
    for rating in ratings:
        rating.session_key = ''
        rating.user_id = user
        rating.save()


def counsel_page(request, template_name='counsel_page.html'):
    if request.is_ajax():
        if 'article' in request.GET and 'rating' in request.GET:
            article = request.GET.get('article')
            rating = request.GET.get('rating')
            pic = get_object_or_404(Picture, article=article)
            pic_rating = int(rating)
            if pic_rating in [1, 2, 3, 4, 5]:
                if request.user.is_authenticated():
                    new_rating = Rating(picture_id=pic.id, rating=pic_rating, user_id=request.user.id)
                    new_rating.save()
                else:
                    new_rating = Rating(picture_id=pic.id, rating=pic_rating, session_key=request.session.session_key)
                    new_rating.save()
                return HttpResponse('')
            else:
                raise Http404
        else:
            raise Http404
    else:
        param = {}
        param = check_cart(request, param)
        param.update(csrf(request))
        param['user_login_form'] = UserForm
        param['rating_email'] = RatingPersonEmailForm
        if request.user.is_authenticated():
            rated_pics = Rating.objects.filter(user=request.user.id).values_list('picture', flat=True)
        else:
            rated_pics = Rating.objects.filter(session_key=request.session.session_key).values_list('picture',
                                                                                                    flat=True)
        pics_to_rate = Picture.objects.exclude(id__in=rated_pics)
        least_rated = pics_to_rate.annotate(num_rated=Count('rating')).order_by('num_rated').order_by()[:5]
        list_pic = list(pics_to_rate.exclude(id__in=least_rated))
        shuffle(list_pic)
        final_list_pics = list_pic[:10] + list(least_rated)
        param.update({'pics': final_list_pics})
        return render_to_response(template_name, param, context_instance=RequestContext(request))


def save_email_after_rating(request):
    if request.is_ajax() and request.method == 'POST':
        data = RatingPersonEmailForm(request.POST)
        s_data = data.save(commit=False)
        if request.user.is_authenticated():
            s_data.user_id = request.user.id
        else:
            s_data.session_key = request.session.session_key
        s_data.save()
        return HttpResponse(json.dumps({'response': 'OK', 'result': 'success'}))
    else:
        raise Http404
