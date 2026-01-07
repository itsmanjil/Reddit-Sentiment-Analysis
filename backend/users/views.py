from copyreg import constructor
from email import message
from turtle import update
from django.core.exceptions import ValidationError
from django.contrib.auth.hashers import check_password
from django.contrib.auth import login, logout
from rest_framework import serializers
from rest_framework.authtoken.models import Token
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.authtoken.models import Token
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.contrib.auth.decorators import login_required
# from users.authentication import expires_in

from users.models import NewUser
from app.models import TweetAnalysis
from .serializers import RegistrationSerializer
# from .authentication import token_expire_handler, expires_in
from .authentication import ExpiringTokenAuthentication

import pytz
import datetime


# @api_view(['POST',])
# @permission_classes([AllowAny])
# def registration_view(request):
#     print('---', request)

#     if request.method == 'POST':
#         serializer = RegistrationSerializer(data=request.data)
#         data = {}
#         if serializer.is_valid(): 
#             user = serializer.save()  #call the overwritten save method from serializers
#             data['response'] = "Successfully registered new user."
#             data['email'] = user.email
#             data['user_name'] = user.user_name
#             token = Token.objects.get(user=user).key
#             data['token'] = token
#         else:
#             data = serializer.errors        
#         return Response(data)

from django.db import IntegrityError
@api_view(["POST"])
# @permission_classes([AllowAny])
def registration_view(request):
    print('---------')
    data = {}
    serializer = RegistrationSerializer(data=request.data)
    print(serializers)
    print(data)
    if serializer.is_valid():
        print('yes')
        account = serializer.save()
        account.is_active = True
        account.save()
        # token = Token.objects.get_or_create(user=account)[0].key
        # data["message"] = "user registered successfully"
        data["email"] = account.email
        data["user_name"] = account.user_name
        # data["token"] = token
        return Response({'message':'User registered', 'data':data},status=status.HTTP_200_OK)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
 

@api_view(["POST",])
@permission_classes([AllowAny])
def login_view(request):
    print('------------------------req', request.data['email'])
    data = {}
    email = request.data['email']
    password = request.data['password']

    print(email, password)

    try:
        User = NewUser.objects.get(email=email)
        print('user', User.password)
    except BaseException as e:
        raise serializers.ValidationError({"400":f'{str(e)}'})
    
    utc_now = datetime.datetime.utcnow()
    utc_now = utc_now.replace(tzinfo=pytz.utc)

    # Token.objects.filter(user=User, created__lt = utc_now - datetime.timedelta(seconds=settings.TOKEN_EXPIRED_AFTER_SECONDS)).delete()
    # token = Token.objects.get_or_create(user=User)[0].key

    # token = Token.objects.get_or_create(user=User)[0].key
    # is_expired, token = token_expire_handler(token)

    if not check_password(password, User.password):
        raise serializers.ValidationError({"error": "Incorrect login credentials"})

    if User:
        if User.is_active:
            login(request, User)
            data["message"] = "User logged in."
            data["email"] = User.email
            data["id"] = User.id
            data["is_registered"] = User.is_registered
            res = {"data": data, 
            # "token": token
            # , "expires_in":expires_in(token)
            }

            return Response(res)

        else:
            raise ValidationError({"message":"User not active"})
    else:
        raise ValidationError({"message":"Account doesnot exists."})


@api_view(["GET",])
@permission_classes([IsAuthenticated])
def logout_view(request):
#    request.user.auth_token.delete()
   logout(request)
   return Response({"message":"User logged out"})

@api_view(["GET",])
def get_user(request, id):
    print('reqq', request.user, id)
    user = NewUser.objects.get(id=id)
    print('user', user)

    tweetData = TweetAnalysis.objects.filter(user=user).order_by('-id')
    searched_lists = []
    for tweet in tweetData:
        searched_lists.append(tweet.product_name)

    isSameUser = request.user == user
    if user.is_authenticated & isSameUser:
        print('---userr----',user.id)
        if user.id == id:
            return Response({
            "user_name":user.user_name,
            "email":user.email,
            "id":user.id,
            "searched_list": searched_lists
            #    "expires_in": expires_in(request.auth)
            })
    else:
        return Response({
            "message":"You can only get your information"
        })

