import React, { useContext, useEffect, useState, useRef } from "react";
//kun component lai k chainxa tei line   send a data from a component to all child components.

//focusing on emmail in login

//effect ==> run code in every render when data changes
//Hook that allows you to have state variables in functional components reactive value,change the value
import ReactDOM from "react-dom";
import { useNavigate } from "react-router-dom";
import { useCurrentPng } from "recharts-to-png";
import Sidenavbar from "../../Components/Sidenavbar";
import Fixedplugins from "../../Components/Fixedplugins";
import { Link, useLocation } from "react-router-dom";
import axios from "axios";
import FileSaver from "file-saver";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  Bar,
  ComposedChart,
  ScatterChart,
  Scatter,
} from "recharts";
import axiosInstance from "../../axios";
import jwt_decode from "jwt-decode";
import AuthContext from "../../context/AuthContext";

const data = [
  {
    name: "Page A", //year,month ->Time
    positive: 4000,
    negative: 2400,
    neutral: 2400,
  },
  {
    name: "Page B",
    positive: 3000,
    negative: 1398,
    neutral: 2210,
  },
  {
    name: "Page C",
    positive: 2000,
    negative: 9800,
    neutral: 2290,
  },
  {
    name: "Page D",
    positive: 2780,
    negative: 3908,
    neutral: 2000,
  },
  {
    name: "Page E",
    positive: 1890,
    negative: 4800,
    neutral: 2181,
  },
  {
    name: "Page F",
    positive: 2390,
    negative: 3800,
    neutral: 2500,
  },
  {
    name: "Page G",
    positive: 3490,
    negative: 4300,
    neutral: 2100,
  },
  {
    name: "Page H",
    positive: 3490,
    negative: 4300,
    neutral: 2100,
  },
  {
    name: "Page I",
    positive: 3490,
    negative: 4300,
    neutral: 2100,
  },
];
// const country = [
//   {
//     name: "USA",
//     total: 8800,
//     positive: 4000,
//     negative: 2400,
//     neutral: 2400,
//     z: 1,
//   },
//   {
//     name: "Australia",
//     total: 7000,
//     positive: 3000,
//     negative: 1398,
//     neutral: 2210,
//     z: 2,
//   },
//   {
//     name: "Nepal",
//     total: 13000,
//     positive: 2000,
//     negative: 9800,
//     neutral: 2290,
//     z: 3,
//   },
//   {
//     name: "India",
//     total: 9000,
//     positive: 2780,
//     negative: 3908,
//     neutral: 2000,
//     z: 4,
//   },
//   {
//     name: "Germany",
//     total: 9900,
//     positive: 1890,
//     negative: 4800,
//     neutral: 2181,
//     z: 5,
//   },
//   {
//     name: "Canada",
//     total: 10000,
//     positive: 2390,
//     negative: 3800,
//     neutral: 2500,
//     z: 6,
//   },
// ];
let data01 = [
  // { name: "Positive", value: 0 },
  // { name: "Negative", value: 0 },
  // { name: "Neutral", value: 0 },
];
// custom label for pie chart
const COLORS = ["#FF0000", "#0000FF", "#008001"];

const RADIAN = Math.PI / 180;
const renderCustomizedLabel = ({
  cx,
  cy,
  midAngle,
  innerRadius,
  outerRadius,
  percent,
  index,
}) => {
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  return (
    <text
      x={x}
      y={y}
      fill="white"
      textAnchor={x > cx ? "start" : "end"}
      dominantBaseline="central"
    >
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
};
function Dashboard(props) {
  const navigate = useNavigate();
  const { authToken } = useContext(AuthContext);

  function refresh() {}
  function calendar() {}
  const [user, setUser] = useState({});
  const [tweetdata, setTweetdata] = useState();
  const [hourdata, setHourdata] = useState();
  const [fetchedDate, setFetchedDate] = useState();
  const [product_name, setProductName] = useState();
  const [hasSearched, setHasSearched] = useState(false);
  const [grpahState, setGraphState] = useState(false);

  const getData = async () => {
    try {
      const token = localStorage.getItem("authToken");
      const { user_id, user_name } = jwt_decode(token);
      console.log(user_name);
      if (user_id) {
        const userDatas = await axios({
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
          user_name: userDatas.data.user_name,
          email: userDatas.data.email,
        });
        console.log("user", user);

        const tweetData = await axios({
          method: "GET",
          url: `http://127.0.0.1:8000/api/sentiment/get_sentiment_data/`,
          timeout: 1000 * 10,
          validateStatus: (status) => {
            return status < 500;
          },
          data: {
            product_name: product_name,
          },
          headers: {
            Authorization: authToken
              ? "Bearer " + String(authToken.access)
              : null,
            "Content-Type": "application/json",
            accept: "application/json",
          },
        });

        // let dataParsed = JSON.parse(tweetData.data.data.sentiment_data);
        setTweetdata(tweetData.data.data.output_sentiment);
        setHourdata(tweetData.data.data.hour_data);
        setProductName(tweetData.data.data.product_name);
        setFetchedDate(tweetData.data.data.fetched_date);
        setGraphState(tweetData.data.data.graph_data_available);
        console.log("check", tweetData);
        console.log(
          user.user_name,
          "tweetData....",
          tweetData.data.data.hour_data,
          tweetData.data.data.product_name,
          tweetData.data.data.fetched_date,
          tweetData.data.data.output_sentiment
          // JSON.parse(tweetData.data.data.sentiment_data)
        );

        setHasSearched(true);
      }
    } catch (err) {
      console.log(err);
      setHasSearched(false);
    }
  };
  useEffect(() => {
    getData();
  }, []);

  //line graph
  let [getLinePng, { ref: lineRef }] = useCurrentPng();

  const lineDownload = React.useCallback(async () => {
    const png = await getLinePng();
    // console.log("clicked", lineRef);
    if (png) {
      // Download with FileSaver
      FileSaver.saveAs(png, "line-chart.png");
    }
  }, [getLinePng]);

  //pie-chart
  let [getPiePng, { ref: pieRef }] = useCurrentPng();

  const pieDownload = React.useCallback(async () => {
    const png = await getPiePng();
    // console.log("clicked", pieRef);
    if (png) {
      // Download with FileSaver
      FileSaver.saveAs(png, "pie-chart.png");
    }
  }, [getPiePng]);

  const navigateToReport = () => {
    const token = localStorage.getItem("authToken");
    const { user_id, user_name } = jwt_decode(token);
    navigate(`/report/${product_name}`, {
      state: {
        user_name,
        tweetdata,
        hourdata,
        fetchedDate,
        product_name,
      },
    });
  };

  return (
    <>
      <Sidenavbar />
      <main className="main-content position-relative max-height-vh-100 h-100 border-radius-lg ">
        <nav
          className="navbar navbar-main navbar-expand-lg px-0 shadow-none border-radius-xl"
          id="navbarBlur"
          navbar-scroll="true"
        >
          <div className="container-fluid py-1 px-3">
            <nav aria-label="breadcrumb">
              <h2 className="font-weight-bolder mb-0">Dashboard</h2>
            </nav>

            <div
              className="collapse navbar-collapse mt-sm-0 mt-2 me-md-0 me-sm-4"
              id="navbar"
            >
              <div className="ms-md-auto pe-md-3 d-flex align-items-center">
                {/* <div className="dropdown float-lg-end pe-4">
                  <button
                    onClick={refresh}
                    class="fas fa-sync"
                    style={{ border: "none", background: "transparent" }}
                  ></button>
                </div>
                <div className="dropdown float-lg-end pe-4">
                  <button
                    onClick={calendar}
                    class="far fa-calendar"
                    style={{ border: "none", background: "transparent" }}
                  ></button>
                </div> */}

                <div className="input-group input-group-outline">
                  <Link to="/search">
                    <input
                      class="btn btn-light profile-button bg-primary"
                      type="button"
                      value="Search Gadget"
                      style={{
                        color: "white",
                        margin: 0,
                        textTransform: "capitalize",
                      }}
                    ></input>
                  </Link>
                </div>
              </div>
              {hasSearched && (
                <div className="dropdown float-lg-end pe-4">
                  <button
                    onClick={navigateToReport}
                    class="fas fa-file"
                    style={{ border: "none", background: "transparent" }}
                  >
                    <span style={{ marginLeft: "5px" }}>Report</span>
                  </button>
                </div>
              )}
              {/* <div className="input-group input-group-outline">
                  <label className="form-label">Search...</label>
                  <input type="text" className="form-control" />
                </div> */}

              <ul className="navbar-nav  justify-content-end">
                <li className="nav-item d-flex align-items-center">
                  <Link
                    to="/profile"
                    className="nav-link text-body font-weight-bold px-0"
                  >
                    <i className="fa fa-user me-sm-1"></i>
                    <span className="d-sm-inline d-none">{user.user_name}</span>
                  </Link>
                </li>
                <li className="nav-item d-xl-none ps-3 d-flex align-items-center">
                  <a
                    href="javascript:;"
                    className="nav-link text-body p-0"
                    id="iconNavbarSidenav"
                  >
                    <div className="sidenav-toggler-inner">
                      <i className="sidenav-toggler-line"></i>
                      <i className="sidenav-toggler-line"></i>
                      <i className="sidenav-toggler-line"></i>
                    </div>
                  </a>
                </li>
                {/* <li className="nav-item px-3 d-flex align-items-center">
                  <a href="javascript:;" className="nav-link text-body p-0">
                    <i className="fa fa-cog fixed-plugin-button-nav cursor-pointer"></i>
                  </a>
                </li> */}
                {/* <li className="nav-item dropdown pe-2 d-flex align-items-center">
                  <a
                    href="javascript:;"
                    className="nav-link text-body p-0"
                    id="dropdownMenuButton"
                    data-bs-toggle="dropdown"
                    aria-expanded="false"
                  >
                    <i className="fa fa-bell cursor-pointer"></i>
                  </a> */}
                {/* <ul className="dropdown-menu  dropdown-menu-end  px-2 py-3 me-sm-n4" aria-labelledby="dropdownMenuButton">
                                        <li className="mb-2">
                                            <a className="dropdown-item border-radius-md" href="javascript:;">
                                                <div className="d-flex py-1">
                                                    <div className="my-auto">
                                                        <img src="../assets/img/team-2.jpg" className="avatar avatar-sm  me-3 " />
                                                    </div>
                                                    <div className="d-flex flex-column justify-content-center">
                                                        <h6 className="text-sm font-weight-normal mb-1">
                                                            <span className="font-weight-bold">New message</span> from Laur
                                                        </h6>
                                                        <p className="text-xs text-secondary mb-0">
                                                            <i className="fa fa-clock me-1"></i>
                                                            13 minutes ago
                                                        </p>
                                                    </div>
                                                </div>
                                            </a>
                                        </li>
                                        <li className="mb-2">
                                            <a className="dropdown-item border-radius-md" href="javascript:;">
                                                <div className="d-flex py-1">
                                                    <div className="my-auto">
                                                        <img src="../assets/img/small-logos/logo-spotify.svg" className="avatar avatar-sm bg-gradient-dark  me-3 " />
                                                    </div>
                                                    <div className="d-flex flex-column justify-content-center">
                                                        <h6 className="text-sm font-weight-normal mb-1">
                                                            <span className="font-weight-bold">New album</span> by Travis Scott
                                                        </h6>
                                                        <p className="text-xs text-secondary mb-0">
                                                            <i className="fa fa-clock me-1"></i>
                                                            1 day
                                                        </p>
                                                    </div>
                                                </div>
                                            </a>
                                        </li>
                                        <li>
                                            <a className="dropdown-item border-radius-md" href="javascript:;">
                                                <div className="d-flex py-1">
                                                    <div className="avatar avatar-sm bg-gradient-secondary  me-3  my-auto">
                                                    </div>
                                                    <div className="d-flex flex-column justify-content-center">
                                                        <h6 className="text-sm font-weight-normal mb-1">
                                                            Payment successfully completed
                                                        </h6>
                                                        <p className="text-xs text-secondary mb-0">
                                                            <i className="fa fa-clock me-1"></i>
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

        {!hasSearched && <h6 className="mx-3">Nothing to show.</h6>}
        {hasSearched && (
          <div className="container-fluid py-4">
            <div className="row">
              <h6>Search results for {product_name}</h6>
              <div className="col-xl-3 col-sm-6 mb-xl-0 mb-4">
                <div className="card">
                  <div className="card-header p-3 pt-2">
                    {/* <div className="icon icon-lg icon-shape bg-gradient-dark shadow-dark text-center border-radius-xl mt-n4 position-absolute">
                                        <i className="material-icons opacity-10">weekend</i>
                                    </div> */}
                    <div className="text-left pt-1">
                      <p className="text-sm mb-0 text-capitalize">
                        Total Reviews
                      </p>
                      <h4 className="mb-0">
                        {tweetdata[0].value +
                          tweetdata[1].value +
                          tweetdata[2].value}
                      </h4>
                      {/* pull total reviews data into this h4 */}
                    </div>
                  </div>
                  <hr className="dark horizontal my-0" />
                  {/* <div className="card-footer p-3">
                                    <p className="mb-0"><span className="text-success text-sm font-weight-bolder">+55% </span>than lask week</p>
                                </div> */}
                </div>
              </div>
              <div className="col-xl-3 col-sm-6 mb-xl-0 mb-4">
                <div className="card">
                  <div className="card-header p-3 pt-2">
                    {/* <div className="icon icon-lg icon-shape bg-gradient-dark shadow-dark text-center border-radius-xl mt-n4 position-absolute">
                                        <i className="material-icons opacity-10">weekend</i>
                                    </div> */}
                    <div className="text-left pt-1">
                      <p className="text-sm mb-0 text-capitalize">
                        Positive Reviews
                      </p>
                      <h4 className="mb-0">{tweetdata[2].value}</h4>
                      {/* pull positive reviews data into this h4 */}
                    </div>
                  </div>
                  <hr className="dark horizontal my-0" />
                  {/* <div className="card-footer p-3">
                                    <p className="mb-0"><span className="text-success text-sm font-weight-bolder">+55% </span>than lask week</p>
                                </div> */}
                </div>
              </div>
              <div className="col-xl-3 col-sm-6 mb-xl-0 mb-4">
                <div className="card">
                  <div className="card-header p-3 pt-2">
                    {/* <div className="icon icon-lg icon-shape bg-gradient-dark shadow-dark text-center border-radius-xl mt-n4 position-absolute">
                                        <i className="material-icons opacity-10">weekend</i>
                                    </div> */}
                    <div className="text-left pt-1">
                      <p className="text-sm mb-0 text-capitalize">
                        Negative Reviews
                      </p>
                      <h4 className="mb-0">{tweetdata[0].value}</h4>
                      {/* pull negative reviews data into this h4 */}
                    </div>
                  </div>
                  <hr className="dark horizontal my-0" />
                  {/* <div className="card-footer p-3">
                                    <p className="mb-0"><span className="text-success text-sm font-weight-bolder">+55% </span>than lask week</p>
                                </div> */}
                </div>
              </div>
              <div className="col-xl-3 col-sm-6">
                <div className="card">
                  <div className="card-header p-3 pt-2">
                    {/* <div className="icon icon-lg icon-shape bg-gradient-info shadow-info text-center border-radius-xl mt-n4 position-absolute">
                                        <i className="material-icons opacity-10">weekend</i>
                                    </div> */}
                    <div className="text-left pt-1">
                      <p className="text-sm mb-0 text-capitalize">
                        Neutral Reviews
                      </p>
                      <h4 className="mb-0">{tweetdata[1].value}</h4>
                    </div>
                  </div>
                  <hr className="dark horizontal my-0" />
                  {/* <div className="card-footer p-3">
                                    <p className="mb-0"><span className="text-success text-sm font-weight-bolder">+5% </span>than yesterday</p>
                                </div> */}
                </div>
              </div>
            </div>
            <br />
            {hourdata && (
              <div className="row mb-4">
                <div className="col-lg-8 col-md-6 mb-md-0 mb-4">
                  <div className="card">
                    <div className="card-header pb-0">
                      <div className="row">
                        <div className="col-lg-6 col-7">
                          <p>Customer Sentiment (on {fetchedDate})</p>
                          {!grpahState && (
                            <span style={{ color: "red" }}>
                              No fetched data for on {fetchedDate}
                            </span>
                          )}
                        </div>
                        <div className="col-lg-6 col-5 my-auto text-end">
                          <div className="dropdown float-lg-end">
                            <button
                              onClick={lineDownload}
                              class="fas fa-download"
                              style={{
                                margin: "15px",
                                border: "none",
                                background: "transparent",
                              }}
                            ></button>
                            {/* <button
                              onClick={refresh}
                              class="fas fa-sync"
                              style={{
                                border: "none",
                                background: "transparent",
                              }}
                            ></button> */}
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="card-body px-0 pb-2">
                      <div className="chart">
                        <ResponsiveContainer width="100%" height={370}>
                          <LineChart
                            ref={lineRef}
                            id="chart-bars"
                            className="chart-canvas"
                            width={700}
                            height={320}
                            data={hourdata}
                            margin={{
                              top: 5,
                              right: 30,
                              left: 20,
                              bottom: 5,
                            }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                              dataKey="time"
                              label={{
                                value: "No. of Tweets",
                                position: "bottom",
                              }}
                              sclaeToFit="true"
                              verticalAnchor="start"
                              textAnchor="end"
                              interval={0}
                              angle="-18"
                              tick={{ fontSize: "12px" }}
                              // axisLine={false}
                            />
                            <YAxis
                              label={{
                                value: "No. of Tweets",
                                angle: -90,
                                position: "insideLeft",
                              }}
                            />
                            <Tooltip />
                            <Legend
                              verticalAlign="top"
                              align="right"
                              wrapperStyle={{
                                left: 0,
                                top: -20,
                                paddingBottom: -60,
                              }}
                            />
                            <Line
                              type="monotone"
                              dataKey="positive"
                              stroke="#008001"
                              activeDot={{ r: 8 }}
                            />
                            <Line
                              type="monotone"
                              dataKey="negative"
                              stroke="#FF0000"
                              activeDot={{ r: 8 }}
                            />
                            <Line
                              type="monotone"
                              dataKey="neutral"
                              stroke="#0000FF"
                              activeDot={{ r: 8 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                        <p
                          style={{
                            textAlignVertical: "center",
                            textAlign: "center",
                          }}
                        >
                          Hours
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="col-lg-4 col-md-6">
                  <div className="card h-100">
                    <div className="card-header pb-0">
                      <div className="row">
                        <div className="col-lg-6 col-7">
                          <h6>Types of Emotion</h6>
                        </div>
                        <div className="col-lg-6 col-5 my-auto text-end">
                          <div className="float-lg-end">
                            <button
                              onClick={pieDownload}
                              class="fas fa-download"
                              style={{
                                margin: "15px",
                                border: "none",
                                background: "transparent",
                              }}
                            ></button>
                            {/* <button
                              onClick={refresh}
                              class="fas fa-sync"
                              style={{
                                border: "none",
                                background: "transparent",
                              }}
                            ></button> */}
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="card-body p-3">
                      <div className="chart">
                        <ResponsiveContainer width="100%" height={250}>
                          {tweetdata && (
                            <PieChart width={200} height={250} ref={pieRef}>
                              <Pie
                                dataKey="value"
                                isAnimationActive={false}
                                data={tweetdata}
                                outerRadius={80}
                                labelLine={false}
                                fill="#8884d8"
                                label={renderCustomizedLabel}
                              >
                                {data.map((entry, index) => (
                                  <Cell
                                    key={`cell-${index}`}
                                    fill={COLORS[index % COLORS.length]}
                                  />
                                ))}
                              </Pie>

                              <Tooltip />
                            </PieChart>
                          )}
                        </ResponsiveContainer>
                        <div>
                          <ul>
                            <ol>
                              <i
                                className="fas fa-circle"
                                style={{ color: "#008001" }}
                              ></i>{" "}
                              : Positive
                            </ol>
                            <ol>
                              <i
                                className="fas fa-circle"
                                style={{ color: "#FF0000" }}
                              ></i>{" "}
                              : Negative
                            </ol>
                            <ol>
                              <i
                                className="fas fa-circle"
                                style={{ color: "#0000FF" }}
                              ></i>{" "}
                              : Neutral
                            </ol>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            {/* <div className="row mb-4">
                        <div className="col-lg-8 col-md-6 mb-md-0 mb-4">
                            <div className="card">
                                <div className="card-header pb-0">
                                    <div className="row">
                                        <div className="col-lg-6 col-7">
                                            <h6>Country wise breakdown</h6>
                                        </div>
                                        <div className="col-lg-6 col-5 my-auto text-end">
                                            <div className="dropdown float-lg-end pe-4">
                                                <button onClick={download} class="fas fa-download" style={{ margin: '15px', border: 'none', background: 'transparent' }}></button>
                                                <button onClick={refresh} class="fas fa-sync" style={{ border: 'none', background: 'transparent' }}></button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div className="card-body px-0 pb-2">
                                    <div className="chart" style={{ width: '100%', height: 300 }}>
                                        <ResponsiveContainer>
                                            <ScatterChart width={400}
                                                height={400}
                                                margin={{
                                                    top: 20,
                                                    right: 20,
                                                    bottom: 20,
                                                    left: 20,
                                                    data: {country}
                                                }}>
                                                <CartesianGrid />
                                                <XAxis type="number" dataKey="total" name="Number of reviews" />
                                                <YAxis type="number" dataKey="z" name="Country"/>
                                                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                                                <Scatter name="A school" data={data} fill="#8884d8">
                                                </Scatter>
                                            </ScatterChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="col-lg-4 col-md-6">
                            <div className="card h-100">
                                <div className="card-header pb-0">
                                    <div className="row">
                                        <div className="col-lg-6 col-7">
                                            <h6>(something)</h6>
                                        </div>
                                        <div className="col-lg-6 col-5 my-auto text-end">
                                            <div className="float-lg-end">
                                                <button onClick={download} class="fas fa-download" style={{ margin: '15px', border: 'none', background: 'transparent' }}></button>
                                                <button onClick={refresh} class="fas fa-sync" style={{ border: 'none', background: 'transparent' }}></button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div className="card-body p-3">
                                </div>
                            </div>
                        </div>
                    </div> */}
          </div>
        )}
      </main>
      <Fixedplugins />
    </>
  );
}

export default Dashboard;