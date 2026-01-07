import React, { useContext, useEffect, useState } from "react";
import ReactDOM from "react-dom";
import { useNavigate, useLocation } from "react-router-dom";
import Navbar from "../../Components/Navbar";
import Sidenavbar from "../../Components/Sidenavbar";
import Fixedplugins from "../../Components/Fixedplugins";
import { Link } from "react-router-dom";
import axiosInstance from "../../axios";
import jwt_decode from "jwt-decode";
import axios from "axios";
import AuthContext from "../../context/AuthContext";

function Profile(props) {
  // const { id } = props.match.params;
  // console.log(id);
  const { authToken } = useContext(AuthContext);
  const [user, setUser] = useState({});
  const [searchedList, setSearchedList] = useState([]);
  const getData = async () => {
    try {
      const token = localStorage.getItem("authToken");
      const { user_id } = jwt_decode(token);
      console.log(user_id);
      if (user_id) {
        try {
          const userData = await axios({
            method: "GET",
            url: `http://127.0.0.1:8000/api/user/me/${user_id}`,
            timeout: 1000 * 10,
            validateStatus: (status) => {
              return status < 500;
            },
            headers: {
              Authorization: authToken
                ? "Bearer " + String(authToken.access)
                : null,
              "Content-Type": "application/json",
              accept: "application/json",
            },
          });
          setUser({
            user_name: userData.data.user_name,
            email: userData.data.email,
          });
          setSearchedList(userData.data.searched_list);
          console.log("user", user, userData);
        } catch (e) {
          console.log(e);
        }
      }
    } catch (err) {
      console.log(err.message);
    }
  };
  useEffect(() => {
    getData();
  }, []);
  return (
    <>
      <Sidenavbar />
      <div class="main-content position-relative bg-gray-100 max-height-vh-100 h-100">
        <nav
          class="navbar navbar-main navbar-expand-lg px-0 mx-3 shadow-none border-radius-xl"
          id="navbarBlur"
          navbar-scroll="true"
        >
          <div class="container-fluid py-1 px-3">
            <nav aria-label="breadcrumb">
              <ol class="breadcrumb bg-transparent mb-0 pb-0 pt-1 px-0 me-sm-6 me-5">
                <li
                  class="breadcrumb-item text-sm text-dark active"
                  aria-current="page"
                ></li>
              </ol>
              <h2 class="font-weight-bolder mb-0">Profile</h2>
            </nav>
            <div
              class="collapse navbar-collapse mt-sm-0 mt-2 me-md-0 me-sm-4"
              id="navbar"
            >
              <div class="ms-md-auto pe-md-3 d-flex align-items-center">
                <div className="input-group input-group-outline">
                  <Link to="/search">
                    <input
                      class="btn btn-light profile-button bg-primary"
                      type="button"
                      value="Search Gadget"
                      style={{
                        margin: 0,
                        textTransform: "capitalize",
                        color: "white",
                      }}
                    ></input>
                  </Link>
                </div>
              </div>
              <ul class="navbar-nav  justify-content-end">
                <li class="nav-item d-flex align-items-center">
                  <a
                    href="javascript:;"
                    class="nav-link text-body font-weight-bold px-0"
                  >
                    <i class="fa fa-user me-sm-1"></i>
                    <span class="d-sm-inline d-none">{user.user_name}</span>
                  </a>
                </li>
                <li class="nav-item d-xl-none ps-3 d-flex align-items-center">
                  <a
                    href="javascript:;"
                    class="nav-link text-body p-0"
                    id="iconNavbarSidenav"
                  >
                    <div class="sidenav-toggler-inner">
                      <i class="sidenav-toggler-line"></i>
                      <i class="sidenav-toggler-line"></i>
                      <i class="sidenav-toggler-line"></i>
                    </div>
                  </a>
                </li>
                {/*<li class="nav-item px-3 d-flex align-items-center">
                  <a href="javascript:;" class="nav-link text-body p-0">
                    <i class="fa fa-cog fixed-plugin-button-nav cursor-pointer"></i>
                  </a>
                </li> */}
                {/* <li class="nav-item dropdown pe-2 d-flex align-items-center">
                  <a
                    href="javascript:;"
                    class="nav-link text-body p-0"
                    id="dropdownMenuButton"
                    data-bs-toggle="dropdown"
                    aria-expanded="false"
                  >
                    <i class="fa fa-bell cursor-pointer"></i>
                  </a> */}
                {/* <ul class="dropdown-menu  dropdown-menu-end  px-2 py-3 me-sm-n4" aria-labelledby="dropdownMenuButton">
                                        <li class="mb-2">
                                            <a class="dropdown-item border-radius-md" href="javascript:;">
                                                <div class="d-flex py-1">
                                                    <div class="my-auto">
                                                        <img src="../assets/img/team-2.jpg" class="avatar avatar-sm  me-3 " />
                                                    </div>
                                                    <div class="d-flex flex-column justify-content-center">
                                                        <h6 class="text-sm font-weight-normal mb-1">
                                                            <span class="font-weight-bold">New message</span> from Laur
                                                        </h6>
                                                        <p class="text-xs text-secondary mb-0">
                                                            <i class="fa fa-clock me-1"></i>
                                                            13 minutes ago
                                                        </p>
                                                    </div>
                                                </div>
                                            </a>
                                        </li>
                                        <li class="mb-2">
                                            <a class="dropdown-item border-radius-md" href="javascript:;">
                                                <div class="d-flex py-1">
                                                    <div class="my-auto">
                                                        <img src="../assets/img/small-logos/logo-spotify.svg" class="avatar avatar-sm bg-gradient-dark  me-3 " />
                                                    </div>
                                                    <div class="d-flex flex-column justify-content-center">
                                                        <h6 class="text-sm font-weight-normal mb-1">
                                                            <span class="font-weight-bold">New album</span> by Travis Scott
                                                        </h6>
                                                        <p class="text-xs text-secondary mb-0">
                                                            <i class="fa fa-clock me-1"></i>
                                                            1 day
                                                        </p>
                                                    </div>
                                                </div>
                                            </a>
                                        </li>
                                        <li>
                                            <a class="dropdown-item border-radius-md" href="javascript:;">
                                                <div class="d-flex py-1">
                                                    <div class="avatar avatar-sm bg-gradient-secondary  me-3  my-auto">
                                                        <svg width="12px" height="12px" viewBox="0 0 43 36" version="1.1" xmlns="http://www.w3.org/2000/svg">
                                                            <title>credit-card</title>
                                                            <g stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
                                                                <g transform="translate(-2169.000000, -745.000000)" fill="#FFFFFF" fill-rule="nonzero">
                                                                    <g transform="translate(1716.000000, 291.000000)">
                                                                        <g transform="translate(453.000000, 454.000000)">
                                                                            <path class="color-background" d="M43,10.7482083 L43,3.58333333 C43,1.60354167 41.3964583,0 39.4166667,0 L3.58333333,0 C1.60354167,0 0,1.60354167 0,3.58333333 L0,10.7482083 L43,10.7482083 Z" opacity="0.593633743"></path>
                                                                            <path class="color-background" d="M0,16.125 L0,32.25 C0,34.2297917 1.60354167,35.8333333 3.58333333,35.8333333 L39.4166667,35.8333333 C41.3964583,35.8333333 43,34.2297917 43,32.25 L43,16.125 L0,16.125 Z M19.7083333,26.875 L7.16666667,26.875 L7.16666667,23.2916667 L19.7083333,23.2916667 L19.7083333,26.875 Z M35.8333333,26.875 L28.6666667,26.875 L28.6666667,23.2916667 L35.8333333,23.2916667 L35.8333333,26.875 Z"></path>
                                                                        </g>
                                                                    </g>
                                                                </g>
                                                            </g>
                                                        </svg>
                                                    </div>
                                                    <div class="d-flex flex-column justify-content-center">
                                                        <h6 class="text-sm font-weight-normal mb-1">
                                                            Payment successfully completed
                                                        </h6>
                                                        <p class="text-xs text-secondary mb-0">
                                                            <i class="fa fa-clock me-1"></i>
                                                            2 days
                                                        </p>
                                                    </div>
                                                </div>
                                            </a>
                                        </li>
                                    </ul> */}
                {/* </li> */}
              </ul>
            </div>
          </div>
        </nav>
        <div class="container-fluid px-2 px-md-4">
          <div class="page-header min-height-300 border-radius-xl mt-4">
            {" "}
            {/*style="background-image: url('https://images.unsplash.com/photo-1531512073830-ba890ca4eba2?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80');"> */}
            <span class="mask  bg-primary  opacity-6"></span>
          </div>
          <div class="card card-body mx-3 mx-md-4 mt-n6">
            <div class="row gx-4 mb-2">
              <div class="col-auto">
                <div class="avatar avatar-xl position-relative">
                  <img
                    src="../assets/img/bruce-mars.jpg"
                    alt="profile_image"
                    class="w-100 border-radius-lg shadow-sm"
                  />
                </div>
              </div>
              <div class="col-auto my-auto">
                <div class="h-100">
                  <h5 class="mb-1" style={{ "text-transform": "uppercase" }}>
                    {user.user_name}
                  </h5>
                  <p class="mb-0 font-weight-normal text-sm">
                    {/* CEO / Co-Founder */}
                    {user.email}
                  </p>
                </div>
              </div>
              <div class="col-lg-4 col-md-6 my-sm-auto ms-sm-auto me-sm-0 mx-auto mt-3">
                <div class="nav-wrapper position-relative end-0">
                  <ul class="nav nav-pills nav-fill p-1" role="tablist">
                    <li class="nav-item">
                      <Link
                        to="/"
                        class="nav-link mb-0 px-0 py-1 active "
                        aria-selected="true"
                      >
                        <i class="material-icons text-lg position-relative">
                          home
                        </i>
                        <span class="ms-1">App</span>
                      </Link>
                    </li>
                    {/* <li class="nav-item">
                      <a
                        class="nav-link mb-0 px-0 py-1 "
                        data-bs-toggle="tab"
                        href="javascript:;"
                        role="tab"
                        aria-selected="false"
                      >
                        <i class="material-icons text-lg position-relative">
                          email
                        </i>
                        <span class="ms-1">Messages</span>
                      </a>
                    </li> */}
                    {/* <li class="nav-item">
                      <a
                        class="nav-link mb-0 px-0 py-1 "
                        data-bs-toggle="tab"
                        href="javascript:;"
                        role="tab"
                        aria-selected="false"
                      >
                        <i class="material-icons text-lg position-relative">
                          settings
                        </i>
                        <span class="ms-1">Settings</span>
                      </a>
                    </li> */}
                  </ul>
                </div>
              </div>
            </div>
            <div class="row">
              <div class="row">
                <div class="col-12 col-xl-4">
                  <div class="card card-plain h-100">
                    <div class="card-header pb-0 p-3">
                      <div class="row">
                        <div class="col-md-8 d-flex align-items-center">
                          <h6 class="mb-0">Profile Information</h6>
                        </div>
                        <div class="col-md-4 text-end">
                          <Link to="/editprofile">
                            <i
                              class="fas fa-user-edit text-secondary text-sm"
                              data-bs-toggle="tooltip"
                              data-bs-placement="top"
                              title="Edit Profile"
                            ></i>
                          </Link>
                        </div>
                      </div>
                    </div>
                    <div class="card-body p-3">
                      <p class="text-sm">
                        Welcome to your profile {user.user_name}.
                      </p>
                      {/* <hr class="horizontal gray-light my-4" /> */}
                      <ul class="list-group">
                        <li class="list-group-item border-0 ps-0 pt-0 text-sm">
                          <strong class="text-dark">Name: </strong> &nbsp;
                          {user.user_name}
                        </li>
                        <li class="list-group-item border-0 ps-0 text-sm">
                          <strong class="text-dark">Email: </strong> &nbsp;
                          {user.email}
                        </li>
                        {/* <li class="list-group-item border-0 ps-0 text-sm">
                          <strong class="text-dark">Location: </strong> &nbsp;
                          Nepal
                        </li> */}
                        <li class="list-group-item border-0 ps-0 pb-0">
                          <strong class="text-dark text-sm">Social: </strong>{" "}
                          &nbsp;
                          <a
                            class="btn btn-facebook btn-simple mb-0 ps-1 pe-2 py-0"
                            href="www.facebook.com"
                          >
                            <i class="fab fa-facebook fa-lg"></i>
                          </a>
                          <a
                            class="btn btn-twitter btn-simple mb-0 ps-1 pe-2 py-0"
                            href="www.twitter.com"
                          >
                            <i class="fab fa-twitter fa-lg"></i>
                          </a>
                          <a
                            class="btn btn-instagram btn-simple mb-0 ps-1 pe-2 py-0"
                            href="www.instagram.com"
                          >
                            <i class="fab fa-instagram fa-lg"></i>
                          </a>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
                {searchedList.length !== 0 && (
                  <div class="col-12 col-xl-4">
                    <div class="card card-plain h-100">
                      <div class="card-header pb-0 p-3">
                        <h6 class="mb-0">Search History</h6>
                      </div>
                      <div class="card-body p-3">
                        <h6 class="text-uppercase text-body text-xs font-weight-bolder">
                          You searched for:
                        </h6>
                        <ul class="list-group">
                          {searchedList.map((productName) => {
                            return (
                              <li class="list-group-item border-0 ps-0 pt-0 text-sm">
                                <div class="text-dark">{productName} </div>
                              </li>
                            );
                          })}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
      <Fixedplugins />
    </>
  );
}

export default Profile;
