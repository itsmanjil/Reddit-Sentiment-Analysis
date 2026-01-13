import React, { useState, useEffect, useContext } from "react";
import "./report.css";
import { useLocation, Link } from "react-router-dom";
import AuthContext from "../../context/AuthContext";
import Dashboard from "./Dashboard";

export default function Report() {
  const location = useLocation();
  console.log(location);

  const { authToken } = useContext(AuthContext);

  let sentimentData = location.state;
  let user_name = location.state.user_name;
  console.log(sentimentData);

  const [totalComments, setTotalComments] = useState(0);
  const [user, setUser] = useState({});


  const handlePrint = () => {
     window.print()
  }


  useEffect(() => {
    let total = 0;
    sentimentData.sentimentBreakdown.forEach((sentiment) => {
      total += sentiment.value;
    });

    setTotalComments(total);
  }, []);

  return (
    <div className="container-main">
      <div className="container-nav-wrapper d-print-none">
        <div className="dropdown float-rg-end pe-4 d-print-none">
          <Link
            to="/dashboard"
            style={{ textDecoration: "none", color: "black" }}
          >
            <span className="fa fa-arrow-left text-dark" style={{ fontSize: '28px' }}></span>
          </Link>
        </div>

        <div className="dropdown float-lg-end pe-4 d-print-none">
          <button
            onClick={handlePrint}
            className="btn btn-dark"
            // style={{ border: "none", background: "transparent" }}
          >
            <span className="fas fa-print"></span> Print
          </button>
        </div>
      </div>

      <div className="my-5 page" size="A4">
        <div id="pagePrint">
          <div className="p-5" id="printPage">
            <section className="top-content bb d-flex justify-content-between">
              <div className="logo">
                <h2>YouTube Sentiment Report</h2>
                {/* <!-- <img src="logo.png" alt="" className="img-fluid"> --> */}
              </div>
              <div className="top-left">
                <div className="graphic-path">
                  <p>Report</p>
                </div>
                <div className="position-relative">
                  <p>
                    Report no.:<span>001</span>
                  </p>
                </div>
              </div>
            </section>

            <section className="store-user mt-5">
              <div className="col-10">
                <div className="row bb pb-3">
                  <div className="col-7">
                    <p>Video analyzed:</p>
                    <h2>{sentimentData.videoTitle}</h2>
                    {/* <p className="address"> 777 Brockton Avenue, <br/> Abington MA 2351, <br/>Vestavia Hills AL </p> */}
                  </div>
                  <div className="col-5">
                    <p>Searched By:</p>
                    <h2>{user_name}</h2>
                    {/* <p className="address"> email <br/> Abington MA 2351, <br/>Vestavia Hills AL </p> */}
                  </div>
                </div>
                <div className="row extra-info pt-3">
                  <div className="col-7">
                    <p>
                      Total Comments Analyzed: <span>{totalComments}</span>
                    </p>
                  </div>
                  <div className="col-5">
                    <p>
                      {" "}
                      Date: <span>{sentimentData.fetchedDate}</span>
                    </p>
                  </div>
                </div>
              </div>
            </section>

            <section className="product-area mt-4">
              <table className="table table-hover">
                <thead>
                  <tr>
                    <td>Sentiment</td>
                    <td>Total</td>
                  </tr>
                </thead>
                <tbody>
                  {sentimentData.sentimentBreakdown.map((sentiment, index) => {
                    const label =
                      sentiment.sentiment || sentiment.name || `item-${index}`;
                    return (
                      <tr key={`${label}-${index}`}>
                        <td>
                          <div className="media">
                            <div className="media-body">
                              <p className="mt-0 title">
                                {label}
                              </p>
                            </div>
                          </div>
                        </td>
                        <td>{sentiment.value}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </section>

            <section className="balance-info">
              <div className="row">
                <div className="col-12">
                  <p className="m-0 font-weight-bold"> Note: </p>
                  <p>
                    Your analysis for {sentimentData.videoTitle} processed {totalComments} comments.
                    It includes {sentimentData.sentimentBreakdown[2].value} positive,{" "}
                    {sentimentData.sentimentBreakdown[0].value} negative, and{" "}
                    {sentimentData.sentimentBreakdown[1].value} neutral comments.
                  </p>
                </div>
                

                
              </div>
            </section>

            

            
          </div>
        </div>
      </div>
    </div>
  );
}
