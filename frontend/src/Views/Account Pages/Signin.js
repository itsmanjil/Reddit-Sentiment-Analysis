import React from "react";
import { Link, Redirect } from "react-router-dom";
import { HashLink } from "react-router-hash-link";

// import Navbar from "../../Components/Navbar";
import SigninForm from "./SigninForm";

function Signin() {
  const token = localStorage.getItem("authToken");
  console.log("token", token);
  return (
    <>
      {/* Navbar */}
      <nav
        id="navbarExample"
        class="navbar navbar-expand-lg fixed-top"
        aria-label="Main navigation"
      >
        <div class="container">
          {/* <!-- Image Logo --> */}
          <Link to="/" class="navbar-brand logo-image">
            <img
              src="../assets/img/logo2.png"
              alt="alternative"
              style={{ height: "40px", width: "40px" }}
            />
          </Link>
          <Link to="/" class="navbar-brand logo-text">
            Gadget Reviews
          </Link>
          <button
            class="navbar-toggler p-0 border-0"
            type="button"
            id="navbarSideCollapse"
            aria-label="Toggle navigation"
          >
            <span class="navbar-toggler-icon"></span>
          </button>

          <div
            class="navbar-collapse offcanvas-collapse"
            id="navbarsExampleDefault"
          >
            <ul class="navbar-nav ms-auto navbar-nav-scroll">
              <li class="nav-item">
                <Link to="/" class="nav-link active" aria-current="page">
                  Home
                </Link>
              </li>
              {/* <li class="nav-item">
                <HashLink smooth to="#Dashboard" class="nav-link">
                  Dashboard
                </HashLink>
              </li> */}
              {/* <li class="nav-item">
                <HashLink smooth to="#details" class="nav-link" href="#details">
                  Details
                </HashLink>
              </li> */}

              {token !== null && (
                <>
                  <li class="nav-item">
                    <Link to="/dashboard" class="nav-link" aria-current="page">
                      Dashboard
                    </Link>
                  </li>
                  <li class="nav-item">
                    <Link to="/profile" class="nav-link" aria-current="page">
                      Profile
                    </Link>
                  </li>
                </>
              )}
            </ul>
            {token == null && (
              <span class="nav-item">
                <Link to="/signin" class="btn-outline-sm">
                  Log in
                </Link>
              </span>
            )}
          </div>
        </div>
      </nav>
      <header class="ex-header">
        <div class="container">
          <div class="row">
            <div class="col-xl-10 offset-xl-1">
              <h1 class="text-center">Log In</h1>
            </div>
          </div>
        </div>
      </header>

      <SigninForm />
    </>
  );
}

export default Signin;
