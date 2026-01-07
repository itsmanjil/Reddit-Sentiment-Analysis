import React, {useState} from "react";
import { Link } from "react-router-dom";
import { HashLink } from "react-router-hash-link";
// Link for routing to diff pages & Hashlink for smooth transition to different parts of same page

// (document).ready(function(){
//   ('li').click(function() {
//   ("li.active").removeClass("active");
//   (this).addClass('active');
// });

export default function Navbar() {
  const [isActive,setActive] = useState("home")
  
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
                <Link to="/" className={isActive==='home' ? 'nav-link active' : 'nav-link' } aria-current="page"
                onClick={() => setActive("home")}
                value={` ${isActive === "home" ? 'btn__nav-bar-btn active-link' : 'btn__nav-bar-btn'}`}
                >
                  Home
                </Link>
              </li>
              <li class="nav-item">
                <HashLink smooth to="#features" className={isActive==='features' ? 'nav-link active' : 'nav-link' }
                onClick={() => setActive("features")}
                value={` ${isActive === "features" ? 'btn__nav-bar-btn active-link' : 'btn__nav-bar-btn'}`}>
                  Features
                </HashLink>
              </li>
              <li class="nav-item">
                <HashLink smooth to="#details"  href="#details" className={isActive==='details' ? 'nav-link active' : 'nav-link' }
                onClick={() => setActive("details")}
                value={` ${isActive === "features" ? 'btn__nav-bar-btn active-link' : 'btn__nav-bar-btn'}`}>
                  Details
                </HashLink>
              </li>
              <li class="nav-item">
                <HashLink smooth to="#contact" class="nav-link" href="#details" className={isActive==='contact' ? 'nav-link active' : 'nav-link' }
                onClick={() => setActive("contact")}
                value={` ${isActive === "contact" ? 'btn__nav-bar-btn active-link' : 'btn__nav-bar-btn'}`}>
                  Contact
                </HashLink>
              </li>

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
      {/* End Navbar*/}
    </>
  

 
  );
}

// export default Navbar;
