#pragma once
/*
 * Here we define json structure as defined by the NASA's JPL  SBDB (Small-Body DataBase)
 * Definition can be found here: https://ssd-api.jpl.nasa.gov/doc/sbdb.html#object
 *
 * Developed by Georgi Vidinski ©2020
 *
 * More information about the json library can be found here: https://github.com/nlohmann/json#examples
 */

#include <string>
using std::string;

#include <nlohmann/json.hpp>
using nlohmann::json;

namespace sdbd
{
	struct Signature
	{
		string source;
		string version;
	};

	struct OrbitClass
	{
		string name;
		string code;
	};

	struct Object
	{
		/// <summary>
		/// NEO flag (true or false)
		/// </summary>
		bool neo = false;

		/// <summary>
		/// PHA flag (true or false)
		/// </summary>
		bool pha = false;

		/// <summary>
		/// Primary SPK ID
		/// </summary>
		int spkid = 0;

		// TODO: Needs enumerator 'ObjectKind'
		/// <summary>
		/// Object kind code: an=”numbered asteroid”, au=”unnumbered asteroid”, cn=”numbered comet”, cu=”unnumbered comet”
		/// </summary>
		string kind;

		/// <summary>
		/// Orbit solution identifier (e.g., “102”, “K153/2”)
		/// </summary>
		string orbit_id;

		/// <summary>
		/// Full object name
		/// </summary>
		string fullname;

		/// <summary>
		/// Primary designation (e.g., “4”, “73P-C”, “2009 FD”, “1995 O1”)
		/// </summary>
		string des;

		/// <summary>
		/// Designation prefix (e.g., “C”, “P”, “D”); null for asteroids
		/// </summary>
		string prefix;

		/// <summary>
		/// Data object containing orbit class name and code (e.g., “Jupiter-family Comet”, “JFc”)
		/// </summary>
		OrbitClass orbit_class;
	};

	struct ModelPars
	{
		/// <summary>
		/// Index within the list of estimated and considered parameters, or 0 if simply set
		/// </summary>
		int n;

		/// <summary>
		/// “SET”, “EST”, “CON”
		/// </summary>
		string kind;

		/// <summary>
		/// Short-name of model parameter (e.g., “A2”)
		/// </summary>
		string name;

		// TODO: Units Type of 'value' needs to be confirmed. Could be anything.
		/// <summary>
		/// Short-name of model parameter (e.g., “A2”)
		/// </summary>
		double value;

		/// <summary>
		/// Title for parameter
		/// </summary>
		string title;

		/// <summary>
		/// Definition of parameter
		/// </summary>
		string desc;

		/// <summary>
		/// 1-sigma uncertainty in parameter value
		/// </summary>
		double sigma;

		/// <summary>
		/// Units for parameter (if any)
		/// </summary>
		string units;

	};

	struct Elements
	{
		/// <summary>
		/// Value of element
		/// </summary>
		double value;

		/// <summary>
		/// 1-sigma uncertainty in element value
		/// </summary>
		double sigma;

		/// <summary>
		/// Short-name of element
		/// </summary>
		string name;

		/// <summary>
		/// Title for element
		/// </summary>
		string title;

		/// <summary>
		/// Label for element
		/// </summary>
		string label;

		/// <summary>
		/// Units (if any)
		/// </summary>
		string units;
	};

	struct Orbit
	{
		/// <summary>
		/// Source of the orbit solution: JPL, MPC, SAO
		/// </summary>
		string source;

		/// <summary>
		/// Epoch of the covariance (TDB) in Julian day form
		/// </summary>
		double cov_epoch;

		/// <summary>
		/// MOID relative to Jupiter (au)
		/// </summary>
		double moid_jup;

		/// <summary>
		/// Jupiter Tisserand invariant
		/// </summary>
		double t_jup;

		/// <summary>
		/// Orbit condition code (OCC)
		/// </summary>
		int condition_code;

		/// <summary>
		/// Date/Time, UTC, before which the orbit is not valid; typically 'null'
		/// If '?nv-fmt=jd'requested, output is formatted as Julian date. If '?nv-fmt=cd' requested output is formatted as calendar date/time YYYY-MMM-DD hh:mm.
		/// </summary>
		string not_valid_before;

		/// <summary>
		/// Normalized RMS of orbit fit
		/// </summary>
		double rms;

		/// <summary>
		/// Array of data objects for OD model parameters (if any; see “model_pars” subsection below)
		/// </summary>
		std::vector<ModelPars> model_pars;

		/// <summary>
		/// Orbit solution identifier; in most cases, the JPL solution number; for short-period comets, prefixed by a perihelion identifier
		/// </summary>
		int orbit_id;

		/// <summary>
		/// Name of the orbit producer (if any)
		/// </summary>
		string producer;

		/// <summary>
		/// Date of the first (earliest) observation used in the fit (YYYY-MM-DD where DD and/or MM may be ?? when the day and/or month is not known; e.g., 1919-??-??)
		/// </summary>
		string first_obs;

		/// <summary>
		/// Date/Time of the orbit solution (pacific local time in format YYYY-MM-DD hh:mm:ss)
		/// </summary>
		string soln_date;

		/// <summary>
		/// Flag indicating simple 2-body model was used in the fit
		/// </summary>
		bool two_body;

		/// <summary>
		/// Epoch of osculation (TDB) in Julian day form
		/// </summary>
		double epoch;

		/// <summary>
		/// Equinox of the reference system (e.g., “J2000”)
		/// </summary>
		string equinox;

		/// <summary>
		/// Number of days spanned by the observations used in the fit
		/// </summary>
		int data_arc;

		/// <summary>
		/// Date/Time, UTC, after which the orbit is not valid; typically null.
		/// If '?nv-fmt=jd'requested, output is formatted as Julian date. If '?nv-fmt=cd' requested output is formatted as calendar date/time YYYY-MMM-DD hh:mm.
		/// </summary>
		string not_valid_after;

		/// <summary>
		/// Number of radar delay observations used
		/// </summary>
		int n_del_obs_used;

		/// <summary>
		/// Name of the JPL small-body perturber ephemeris used
		/// </summary>
		string sb_used;

		/// <summary>
		/// Total number of observations used (optical and radar)
		/// </summary>
		int n_obs_used;

		/// <summary>
		/// Comments related to this orbit
		/// </summary>
		string comment;

		/// <summary>
		/// Name of the JPL planetary ephemeris used
		/// </summary>
		string pe_used;

		/// <summary>
		/// Date of the last (latest) observation used in the fit (YYYY-MM-DD where DD and/or MM may be ?? when the day and/or month is not known; e.g., 1919-??-??)
		/// </summary>
		string last_obs;

		/// <summary>
		/// MOID relative to Earth (au)
		/// </summary>
		double moid;

		/// <summary>
		/// Number of radar Doppler observations used
		/// </summary>
		int n_dop_obs_used;

		/// <summary>
		/// Array of data objects for osculating orbital elements
		/// </summary>
		std::vector<Elements> elements;
		//Elements elements[];

		/*static Elements from_json(const json& j)
		{
			Elements elements;
			elements.
		}*/
	};

	struct Asteroid
	{
		Signature signature;
		Object object;
		Orbit orbit;
	};

	void from_json(const json& j, Signature& signature)
	{
		j.at("source").get_to(signature.source);
		j.at("version").get_to(signature.version);
	}

	void from_json(const json& j, OrbitClass& orbitClass)
	{
		j.at("name").get_to(orbitClass.name);
		j.at("code").get_to(orbitClass.code);
	}

	void from_json(const json& j, Object& object)
	{
		if (j.is_null())
		{
			return;
		}

		j.at("neo").get_to(object.neo);
		j.at("pha").get_to(object.pha);

		j.at("orbit_id").get_to(object.orbit_id);
		j.at("kind").get_to(object.kind);
		j.at("fullname").get_to(object.fullname);
		j.at("des").get_to(object.des);

		object.spkid = std::stoi(j.at("spkid").get<string>());

		j.at("prefix").is_null()
			? object.prefix = ""
			: j.at("prefix").get_to(object.prefix);

		object.orbit_class = j.at("orbit_class").get<OrbitClass>();
	}

	void from_json(const json& j, ModelPars& modelPars)
	{
		j.at("n").get_to(modelPars.n);
		j.at("kind").get_to(modelPars.kind);
		j.at("name").get_to(modelPars.name);
		j.at("value").get_to(modelPars.value);
		j.at("title").get_to(modelPars.title);
		j.at("desc").get_to(modelPars.desc);
		j.at("sigma").get_to(modelPars.sigma);
		j.at("units").get_to(modelPars.units);
	}

	void from_json(const json& j, Elements& elements)
	{
		elements.value = std::stod(j.at("value").get<string>());
		elements.sigma = std::stod(j.at("sigma").get<string>());
		j.at("name").get_to(elements.name);
		j.at("title").get_to(elements.title);
		j.at("label").get_to(elements.label);

		j.at("units").is_null()
			? elements.units = ""
			: j.at("units").get_to(elements.units);
	}

	/// <summary>
	/// Usage:
	///		Test my_array[N];
	///		from_json(json, my_array);
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <param name="j"></param>
	/// <param name="t"></param>
	template <typename T, size_t N>
	void from_json(const json& j, T(&t)[N]) {
		if (j.size() > N) {
			throw std::runtime_error("JSON array too large");
		}
		size_t index = 0;
		for (auto& item : j) {
			from_json(item, t[index++]);
		}
	}

	void from_json(const json& j, Orbit& orbit)
	{
		if (j.is_null())
		{
			return;
		}

		j.at("source").get_to(orbit.source);
		orbit.cov_epoch = std::stod(j.at("cov_epoch").get<string>());
		orbit.moid_jup = std::stod(j.at("moid_jup").get<string>());
		orbit.t_jup = std::stod(j.at("t_jup").get<string>());
		orbit.condition_code = std::stoi(j.at("condition_code").get<string>());

		j.at("not_valid_before").is_null()
			? orbit.not_valid_before = ""
			: j.at("not_valid_before").get_to(orbit.not_valid_before);

		orbit.rms = std::stod(j.at("rms").get<string>());
		orbit.orbit_id = std::stoi(j.at("orbit_id").get<string>());
		j.at("producer").get_to(orbit.producer);
		j.at("first_obs").get_to(orbit.first_obs);
		j.at("soln_date").get_to(orbit.soln_date);

		j.at("two_body").is_null()
			? orbit.two_body = false
			: j.at("two_body").get_to(orbit.two_body);

		orbit.epoch = std::stod(j.at("epoch").get<string>());
		j.at("equinox").get_to(orbit.equinox);
		orbit.data_arc = std::stoi(j.at("data_arc").get<string>());

		j.at("not_valid_after").is_null()
			? orbit.not_valid_after = ""
			: j.at("not_valid_after").get_to(orbit.not_valid_after);

		j.at("n_del_obs_used").is_null()
			? orbit.n_del_obs_used = 0
			: orbit.n_del_obs_used = std::stoi(j.at("n_del_obs_used").get<string>());

		j.at("sb_used").get_to(orbit.sb_used);

		j.at("n_obs_used").is_null()
			? orbit.n_obs_used = 0
			: orbit.n_obs_used = std::stoi(j.at("n_obs_used").get<string>());

		j.at("comment").is_null()
			? orbit.comment = ""
			: j.at("comment").get_to(orbit.comment);

		j.at("pe_used").get_to(orbit.pe_used);
		j.at("last_obs").get_to(orbit.last_obs);

		orbit.moid = std::stod(j.at("moid").get<string>());

		j.at("n_dop_obs_used").is_null()
			? orbit.n_dop_obs_used
			: orbit.n_dop_obs_used = std::stoi(j.at("n_dop_obs_used").get<string>());

		auto j_elements = j.at("elements");
		auto e_size = j_elements.size();
		if (e_size > 0)
		{
			std::vector<Elements> elements(e_size);
			size_t index = 0;
			for (auto& item : j_elements)
			{
				from_json(item, elements[index++]);
			}

			auto it = elements.begin();
			orbit.elements.assign(it, elements.end());
		}

		auto j_model_pars = j.at("model_pars");
		auto mp_size = j_model_pars.size();
		if(mp_size > 0)
		{
			std::vector<ModelPars> model_pars(mp_size);
			size_t index = 0;
			for(auto& item : j_model_pars)
			{
				from_json(item, model_pars[index++]);
			}

			auto it = model_pars.begin();
			orbit.model_pars.assign(it, model_pars.end());
		}


		//for (json::iterator it = elements.begin(); it != elements.end(); ++it)
		//{
		//	auto element = (*it).get<Elements>();
		//	auto t = element.at("value").get_to()
		//
		//	//orbit.elements.assign(element);
		//	//std::cerr << *it << '\n';
		//}

		//j.at("model_pars").get_to(orbit.model_pars);
	}



	void from_json(const json& j, Asteroid& asteroid)
	{
		j.at("signature").get_to(asteroid.signature);
		j.at("object").get_to(asteroid.object);
		j.at("orbit").get_to(asteroid.orbit);
	}
}